# Bank Marketing EDA App

A simple, interactive **Exploratory Data Analysis (EDA)** project for the bank
marketing dataset, built with:

- **NumPy** — numeric operations
- **Pandas** — data loading & manipulation
- **Matplotlib / Seaborn** — static plots
- **Streamlit** — interactive web app

No machine learning or feature engineering is included — this is pure data
exploration.

## Project structure

```
eda_app/
├── app.py             # Streamlit application
├── bank.csv           # Dataset
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually https://daman96k--bank-marketing-campaign-eda-project-loknvf.streamlit.app/).

## Features

The app is organized into 5 tabs:

1. **Overview** — shape, dtypes, missing values, descriptive statistics
2. **Univariate** — histograms/boxplots for numeric columns, bar/pie charts
   for categorical columns
3. **Relationships** — numeric vs numeric (scatter + correlation), numeric vs
   categorical (boxplot), categorical vs categorical (stacked bar + cross-tab)
4. **Correlation** — correlation heatmap with selectable method (Pearson,
   Spearman, Kendall)
5. **Target Analysis** — subscription rate (`deposit`) broken down by
   categorical and numeric features

A sidebar lets you filter the dataset by job, marital status, education,
deposit outcome, and age range — all charts and tables update live.

## Dataset

`bank.csv` contains 11,162 rows and 17 columns describing a bank's direct
marketing campaign (phone calls), with the target column `deposit`
indicating whether the client subscribed to a term deposit.
