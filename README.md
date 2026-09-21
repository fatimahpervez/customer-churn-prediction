# Customer Churn Prediction

A deployed machine learning app that predicts whether a telecom customer is likely to churn, built end-to-end from raw data to a live, interactive tool.

**Live app:** https://fatimahpervez-customer-churn-prediction-app-wkjziz.streamlit.app

## The problem

Customer churn is expensive to fix after the fact — retaining an existing customer is cheaper than acquiring a new one. This project identifies which customers are at high risk of leaving *before* they do, so a business could act early with a targeted retention offer.

## Data

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle/IBM) — 7,043 customers, 21 original features covering demographics, account details, and subscribed services.

## Key findings

- **Contract type is the strongest churn driver**: month-to-month customers churn at 42.7%, versus 11.3% for one-year and just 2.8% for two-year contracts.
- Churn is concentrated in low-tenure customers and, separately, in higher monthly bills.
- Fiber optic internet (41.9% churn), missing security/tech-support add-ons (~42% churn each), and electronic check payment (45.3% churn) are all high-risk segments.

## Model

A **Random Forest classifier** (`class_weight='balanced'`) was chosen over a logistic regression baseline. The baseline scored higher raw accuracy (80% vs 76%), but the random forest catches more actual churners (62% recall vs 55%) — a deliberate trade-off, since a missed churner is assumed to cost more than an unnecessary retention offer to someone who was staying anyway.

| Metric | Logistic Regression | Random Forest (final) |
|---|---|---|
| Accuracy | 80% | 76% |
| Churn recall | 55% | 62% |
| Churn precision | 64% | 54% |

## Tech stack

Python, pandas, scikit-learn, Streamlit, deployed on Streamlit Community Cloud.

## Running locally

```
git clone https://github.com/fatimahpervez/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```


## Project structure

```
├── data/ # Raw dataset
├── 01_eda.ipynb # Data cleaning, EDA, model training
├── app.py # Streamlit app
├── churn_model.pkl # Trained model
├── model_columns.pkl # Feature structure for encoding new inputs
└── requirements.txt
```