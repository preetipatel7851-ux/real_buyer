# Machine Learning Based Buyer Segmentation & Investment Profiling for Real Estate Market Intelligence

A Streamlit dashboard that uses K-Means clustering to segment property buyers and explain
their investment motivations, financing patterns and geography.

## Project structure
```
real-estate-buyer-segmentation/
├── app.py               # Streamlit dashboard (4 pages)
├── run_pipeline.py      # cleaning + feature engineering + K-Means + export
├── generate_data.py     # creates sample data (skip if you have your own CSV)
├── data/raw/clients.csv # input dataset
├── data/processed/      # pipeline output used by the dashboard
├── models/              # saved K-Means model
└── requirements.txt
```

## Dataset fields
client_id, client_type, gender, country, region, date_of_birth, acquisition_purpose,
loan_applied, referral_channel, satisfaction_score

## How to run (Windows CMD)
```
pip install -r requirements.txt
python generate_data.py        (only if you don't have your own data/raw/clients.csv)
python run_pipeline.py
streamlit run app.py
```

## Method
1. Clean data: remove duplicate clients, fix dates, fill missing values.
2. Features: age (from date_of_birth), investor flag, loan flag, corporate flag, satisfaction, referral channel.
3. Standardise, then K-Means. k is chosen (4-6) by the best silhouette score.
4. Region/country are deliberately excluded from clustering so the geographic page shows where segments live.
5. Segments are auto-named from their profile (e.g. Corporate Investors, Mortgage Home Buyers).

## Dashboard pages
| Page | What it shows |
|---|---|
| Buyer Segmentation Overview | Cluster distribution, sizes, 2-D segment map |
| Investor Behavior Dashboard | Purpose, financing, channels, satisfaction, age by cluster |
| Geographic Buyer Analysis | World map + regional segment mix |
| Segment Insights Panel | Descriptive statistics and summary per cluster |
