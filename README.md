# Customer Behaviour Analysis & Segmentation Dashboard

## Put these files together
- `app.py`
- `requirements.txt`
- `customer_features.csv`
- `model_comparison.csv`
- `cleaned_purchases.csv`

Copy the three CSV files you downloaded from Google Colab into this folder.

## Run
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run `python -m pip install -r requirements.txt`
4. Run `streamlit run app.py`
5. Open the local URL shown in the terminal (usually http://localhost:8501).

## Pages
- Overview: customer, invoice, revenue and country summaries
- Customer Segmentation: PCA projection and segment profiles
- Purchase Prediction: historical-window repeat-purchase demonstration
- Model Comparison: saved classifier comparison

## Methodology note
The prediction page uses features through 30 September 2011 and a future target window through 29 December 2011. Its displayed metrics use a stratified random split as a baseline; a separate later time-based holdout is recommended for stronger prospective evaluation.
