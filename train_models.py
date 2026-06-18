"""
train_models.py
Builds the customer-segmentation (KMeans) pipeline and the product
recommendation (item-based collaborative filtering) similarity matrix
from online_retail.csv, and saves all artifacts needed by app.py.

Run this once before launching the Streamlit app:
    python train_models.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "online_retail.csv"
MODELS_DIR = "models"


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the raw transaction log and apply the same cleaning rules used
    in the analysis notebook: drop unattributable/cancelled/invalid rows."""
    df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str})

    df_clean = df.dropna(subset=["CustomerID"]).copy()
    df_clean = df_clean[~df_clean["InvoiceNo"].str.startswith("C")]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=["Description"])

    df_clean["Description"] = df_clean["Description"].str.strip()
    df_clean["Country"] = df_clean["Country"].str.strip()
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    return df_clean.reset_index(drop=True)


def build_rfm(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Engineer Recency, Frequency, Monetary features at the customer level."""
    snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df_clean.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


def train_clustering_pipeline(rfm: pd.DataFrame):
    """Log-transform + scale RFM features, fit KMeans (k=4), and derive a
    cluster -> business segment name mapping ranked by R, F, M quality."""
    rfm = rfm.copy()
    rfm["Recency_log"] = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

    X = rfm[["Recency_log", "Frequency_log", "Monetary_log"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=20, max_iter=500, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    profile = rfm.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
    ).reset_index()

    profile["R_rank"] = profile["Recency_mean"].rank(ascending=True)
    profile["F_rank"] = profile["Frequency_mean"].rank(ascending=False)
    profile["M_rank"] = profile["Monetary_mean"].rank(ascending=False)
    profile["avg_rank"] = profile[["R_rank", "F_rank", "M_rank"]].mean(axis=1)
    profile = profile.sort_values("avg_rank").reset_index(drop=True)

    segment_names = ["High-Value", "Regular", "Occasional", "At-Risk"]
    cluster_to_segment = dict(zip(profile["Cluster"], segment_names))

    rfm["Segment"] = rfm["Cluster"].map(cluster_to_segment)

    return scaler, kmeans, cluster_to_segment, rfm


def build_recommendation_matrix(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Build the customer-product matrix and compute item-item cosine similarity."""
    customer_product_matrix = df_clean.pivot_table(
        index="CustomerID", columns="Description", values="Quantity",
        aggfunc="sum", fill_value=0,
    )
    product_customer_matrix = customer_product_matrix.T

    similarity = cosine_similarity(product_customer_matrix.values.astype(np.float32))
    similarity_df = pd.DataFrame(
        similarity, index=product_customer_matrix.index, columns=product_customer_matrix.index,
    ).astype(np.float32)

    return similarity_df


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading and cleaning data...")
    df_clean = load_and_clean_data(DATA_PATH)
    print(f"  -> {len(df_clean):,} clean transaction rows, "
          f"{df_clean['CustomerID'].nunique():,} customers, "
          f"{df_clean['Description'].nunique():,} unique products")

    print("Engineering RFM features...")
    rfm = build_rfm(df_clean)

    print("Training KMeans clustering pipeline...")
    scaler, kmeans, cluster_to_segment, rfm_with_segments = train_clustering_pipeline(rfm)
    print("  -> Segment sizes:")
    print(rfm_with_segments["Segment"].value_counts().to_string())

    print("Building product similarity matrix (this can take a minute)...")
    similarity_df = build_recommendation_matrix(df_clean)
    print(f"  -> Similarity matrix shape: {similarity_df.shape}")

    print("Saving artifacts to disk...")
    joblib.dump(scaler, os.path.join(MODELS_DIR, "rfm_scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    with open(os.path.join(MODELS_DIR, "cluster_segment_map.pkl"), "wb") as f:
        pickle.dump(cluster_to_segment, f)
    similarity_df.to_pickle(os.path.join(MODELS_DIR, "product_similarity.pkl"))

    # Save a sorted list of unique product names for the app's selectbox / suggestions
    product_list = sorted(similarity_df.columns.tolist())
    with open(os.path.join(MODELS_DIR, "product_list.pkl"), "wb") as f:
        pickle.dump(product_list, f)

    # Save segment-level RFM summary stats (used to populate Home page KPIs)
    segment_summary = rfm_with_segments.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    segment_summary["Customer_Count"] = rfm_with_segments["Segment"].value_counts()
    segment_summary.to_pickle(os.path.join(MODELS_DIR, "segment_summary.pkl"))

    print("\nDone. All artifacts saved in the 'models/' folder:")
    for fname in sorted(os.listdir(MODELS_DIR)):
        size_mb = os.path.getsize(os.path.join(MODELS_DIR, fname)) / (1024 ** 2)
        print(f"  models/{fname}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
