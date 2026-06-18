# Shopper Spectrum — Streamlit App

Customer segmentation and product recommendation app for the Shopper Spectrum
capstone project, built on the `online_retail.csv` transaction dataset.

## What's included

- `app.py` — the Streamlit application (Home, Clustering, Recommendation pages)
- `train_models.py` — trains the KMeans segmentation pipeline and the product
  similarity matrix, and saves them to `models/`
- `online_retail.csv` — the raw transaction dataset
- `requirements.txt` — Python dependencies

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `online_retail.csv` is in this same folder (already included).

4. Train the models (only needs to be run once, or whenever the dataset changes):

   ```bash
   python train_models.py
   ```

   This creates a `models/` folder containing:
   - `rfm_scaler.pkl` — fitted StandardScaler for RFM features
   - `kmeans_model.pkl` — fitted KMeans clustering model
   - `cluster_segment_map.pkl` — maps cluster ids to segment names
   - `product_similarity.pkl` — product-to-product cosine similarity matrix
   - `product_list.pkl` — sorted list of all product names
   - `segment_summary.pkl` — average RFM values per segment

5. Launch the app:

   ```bash
   streamlit run app.py
   ```

   Streamlit will print a local URL (typically `http://localhost:8501`) —
   open it in your browser.

## Pages

- **Home** — overview KPIs and segment distribution across the customer base.
- **Clustering** — enter Recency, Frequency, and Monetary values for a
  customer and get an instant segment prediction (High-Value, Regular,
  Occasional, or At-Risk).
- **Recommendation** — enter a product name and get the top 5 most similar
  products based on item-based collaborative filtering (cosine similarity
  over customer purchase quantities).

## Notes

- If you see "Model artifacts not found" when launching the app, run
  `python train_models.py` first.
- The product similarity matrix (`product_similarity.pkl`) is the largest
  artifact (~55-60 MB). If deploying to a platform with strict storage/RAM
  limits, consider retraining `train_models.py` on a trimmed product catalogue
  (e.g. excluding very low-frequency SKUs) to reduce its size.
- To deploy on Streamlit Community Cloud: push this folder to a GitHub repo
  (including the `models/` folder, or add a startup step that runs
  `train_models.py` before the app starts), then point Streamlit Cloud at
  `app.py`.
