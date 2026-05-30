# Pharma Safety Lakehouse

An end-to-end cloud-native data platform for pharmacovigilance analytics. Processes public adverse-event reports from the openFDA API, transforms nested JSON with PySpark, stores curated data in Google BigQuery, and exposes safety insights through an interactive Streamlit dashboard.

---

## Business Problem

Pharmacovigilance teams need to monitor adverse-event reports to identify safety trends, frequent reactions, serious cases, and drug-reaction signals. Raw FAERS data is nested, semi-structured, duplicated, and difficult to query at scale.

This platform answers questions such as:

- Which drugs are most frequently reported?
- Which adverse reactions appear most often?
- Which drug-reaction pairs are potential safety signals?
- Can a machine learning model predict whether a case is serious?

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | Main programming language |
| PySpark | 3.5.1 | Processing and transforming nested datasets |
| Google Cloud Storage | — | Bronze, silver, and gold data layers |
| BigQuery | — | Analytical data warehouse |
| Apache Airflow | 2.9.1 | Pipeline orchestration |
| Docker | — | Reproducible local environment |
| GitLab CI/CD | — | Automated lint, test, and DAG validation |
| pytest | 8.2.0 | Test suite |
| MLflow | 2.13.0 | Experiment tracking |
| scikit-learn | 1.4.2 | Seriousness prediction model |
| Streamlit | 1.35.0 | Interactive analytics dashboard |
| Ruff / Black | — | Code quality and formatting |

---

## Architecture

```
openFDA API
    │
    ▼
Airflow Ingestion DAG
    │
    ▼
GCS Bronze Layer  (raw JSON, partitioned by year/month)
    │
    ▼
PySpark Cleaning Jobs
    │
    ▼
GCS Silver Layer  (clean Parquet tables)
    ├── silver_adverse_events
    ├── silver_drugs
    └── silver_reactions
    │
    ▼
PySpark Aggregation Jobs
    │
    ▼
GCS Gold Layer  (analytics-ready tables)
    ├── gold_safety_overview
    ├── gold_monthly_trends
    ├── gold_drug_summary
    ├── gold_reaction_summary
    ├── gold_drug_reaction_pairs
    └── gold_ml_features
    │
    ├──────────────────────┐
    ▼                      ▼
BigQuery              ML Training
    │                      │
    ▼                      ▼
Streamlit Dashboard   MLflow Tracking
```

---

## Dataset

**Source:** openFDA Drug Adverse Event API — based on FAERS data.

**API endpoint:** `https://api.fda.gov/drug/event.json`

Each raw record contains nested fields:

- Safety report ID and receive date
- Seriousness flags (death, hospitalization, life-threatening)
- Patient demographics (age, sex, country)
- Drug array (name, role, indication)
- Reaction array (MedDRA preferred terms)

---

## Data Model

### Bronze Layer

Raw JSON files stored exactly as downloaded, partitioned by year and month.

```
data/bronze/openfda_events/year=2024/month=01/events.json
```

### Silver Layer

Cleaned and normalized Parquet tables.

**silver_adverse_events**

| Column | Type | Description |
|---|---|---|
| safety_report_id | string | Unique report identifier |
| report_date | date | Parsed from receivedate string |
| country | string | Primary source country |
| is_serious | integer | 1 = serious, 0 = not serious |
| death | integer | 1 = death reported |
| hospitalization | integer | 1 = hospitalization reported |
| patient_sex | string | Male / Female / Unknown |
| patient_age | integer | Age in years, normalized from all openFDA units |

**silver_drugs**

| Column | Type | Description |
|---|---|---|
| safety_report_id | string | Links to adverse_events |
| drug_name | string | Uppercased product name |
| drug_role | string | Suspect / Concomitant / Interacting |
| indication | string | Drug indication if available |

**silver_reactions**

| Column | Type | Description |
|---|---|---|
| safety_report_id | string | Links to adverse_events |
| reaction_name | string | MedDRA preferred term (title case) |

### Gold Layer

| Table | Description |
|---|---|
| gold_safety_overview | Single-row global KPIs |
| gold_monthly_trends | Report counts and seriousness rate by month |
| gold_drug_summary | Per-drug totals and seriousness rate |
| gold_reaction_summary | Per-reaction totals and seriousness rate |
| gold_drug_reaction_pairs | Drug-reaction pair counts for signal detection |
| gold_ml_features | Engineered features for the ML model |

---

## Repository Structure

```
pharma-safety-lakehouse/
├── dags/
│   └── pharmacovigilance_pipeline.py
├── src/
│   ├── ingestion/
│   │   ├── fetch_openfda_events.py
│   │   └── load_to_bigquery.py
│   ├── spark_jobs/
│   │   ├── bronze_to_silver_events.py
│   │   └── build_gold_tables.py
│   ├── quality/
│   │   └── data_quality_checks.py
│   ├── ml/
│   │   ├── train_seriousness_model.py
│   │   ├── evaluate_model.py
│   │   └── export_mlflow_metrics.py
│   ├── dashboard/
│   │   └── streamlit_app.py
│   └── utils/
│       ├── config.py
│       ├── gcp_helpers.py
│       └── spark_session.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_spark_transformations.py
│   ├── test_data_quality.py
│   ├── test_gold_tables.py
│   ├── test_airflow_dag.py
│   └── test_ml_training.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_ml_experimentation.ipynb
├── .gitlab-ci.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.12
- Docker Desktop
- Java 17
- Git
- Windows only: `winutils.exe` and `hadoop.dll` in `C:\hadoop\bin`, `HADOOP_HOME=C:\hadoop`

### Install

```bash
git clone https://github.com/yourusername/pharma-safety-lakehouse.git
cd pharma-safety-lakehouse
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your GCP project ID, bucket name, and service account path.

### Start Docker services

```bash
docker compose -f docker/docker-compose.yml up -d
```

| Service | URL | Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | — |

### Run the pipeline manually

```bash
# Ingest raw data from openFDA
python -m src.ingestion.fetch_openfda_events

# Transform bronze to silver
python -m src.spark_jobs.bronze_to_silver_events

# Build gold tables
python -m src.spark_jobs.build_gold_tables

# Load to BigQuery
python -m src.ingestion.load_to_bigquery

# Train ML model and log to MLflow
python -m src.ml.train_seriousness_model

# Launch dashboard
streamlit run src/dashboard/streamlit_app.py
```

---

## GCP Setup

1. Create a project at https://console.cloud.google.com
2. Enable the BigQuery API and Cloud Storage API
3. Create a service account with roles `BigQuery Admin` and `Storage Admin`
4. Download the JSON key and save it as `service-account.json` in the project root
5. Create a GCS bucket matching your `GCS_BUCKET_NAME` env variable
6. Create two BigQuery datasets: `pharma_safety_silver` and `pharma_safety_gold`

---

## Airflow DAG

The `pharmacovigilance_pipeline` DAG runs monthly on the 1st of each month at 3am.

```
ingest_openfda_data
      │
      ▼
bronze_to_silver
      │
      ▼
data_quality_checks
      │
      ▼
build_gold_tables
      │
      ▼
load_gold_to_bigquery
```

Each task retries twice with a 5-minute delay. The quality check task acts as a gate — if data quality fails, downstream tasks are skipped.

---

## BigQuery

All gold tables are loaded into the `pharma_safety_gold` dataset.

```sql
-- Top drugs by serious reports
SELECT drug_name, total_reports, serious_reports, seriousness_rate_pct
FROM `pharma_safety_gold.drug_summary`
ORDER BY serious_reports DESC
LIMIT 20;

-- Top signal detection pairs
SELECT drug_name, reaction_name, pair_count, seriousness_rate_pct
FROM `pharma_safety_gold.drug_reaction_pairs`
WHERE pair_count >= 10
ORDER BY seriousness_rate_pct DESC
LIMIT 20;
```

---

## Dashboard

Four-page Streamlit dashboard reading from the gold Parquet layer.

| Page | Content |
|---|---|
| Safety Overview | Global KPIs, monthly trend, country distribution, sex distribution |
| Drug Analysis | Top drugs by volume and seriousness rate |
| Reaction Signals | Signal detection scatter plot, top drug-reaction pairs |
| ML Monitoring | Model comparison, metric cards, run history |

```bash
streamlit run src/dashboard/streamlit_app.py
# Opens at http://localhost:8501
```

---

## MLflow

Two models trained and tracked in the `seriousness_prediction` experiment:

- Logistic Regression
- Random Forest

**Features:** patient age, patient sex, country, number of drugs, number of reactions, suspect drug flag.

**Metrics tracked:** accuracy, precision, recall, F1, ROC-AUC.

View at http://localhost:5000 while Docker is running.

---

## Testing

```bash
# Fast tests only (no Spark required)
pytest tests/ -v -m "not spark"

# Spark tests (requires local data files)
pytest tests/ -v -m "spark"

# All tests
pytest tests/ -v
```

| File | Covers |
|---|---|
| test_ingestion.py | API pagination, date filter, file saving |
| test_spark_transformations.py | Silver transformation logic |
| test_data_quality.py | Quality rules pass and fail correctly |
| test_gold_tables.py | Gold table schema and value ranges |
| test_airflow_dag.py | DAG structure, task count, dependencies |
| test_ml_training.py | Feature preparation, metric computation |

---

## CI/CD Pipeline

GitLab CI/CD runs on every push with three stages:

| Stage | What it does |
|---|---|
| lint | Runs Ruff to check for style and import issues |
| unit_tests | Runs fast Python-only tests |
| dag_validation | Imports the Airflow DAG and validates it parses correctly |

Spark tests are excluded from CI and run locally where data files exist.

---

## Key Engineering Decisions

**Binary flag normalization.** openFDA uses 1=yes and 2=no for seriousness flags. The silver transformation converts 2 to 0 so all downstream analytics use consistent binary values.

**Age unit normalization.** openFDA reports patient age in multiple units — years, months, weeks, days, hours, decades. The silver layer converts all to years and rejects biologically impossible values.

**Quality gate in the DAG.** A dedicated data quality task sits between silver and gold. If any check fails the pipeline stops before bad data reaches BigQuery.

**Imports inside Airflow callables.** Heavy imports are inside the callable functions, not at the DAG module level. This prevents import errors from breaking DAG parsing.

---

## Future Improvements

- Incremental ingestion using watermarks instead of full monthly reloads
- Reporting Odds Ratio for formal signal detection
- Terraform infrastructure as code for GCP setup
- Great Expectations for declarative data quality contracts
- dbt for gold layer transformations with lineage tracking
- Model retraining trigger when new monthly data arrives
