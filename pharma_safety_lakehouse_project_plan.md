# Pharma Safety Data Lakehouse

**End-to-End Data Engineering and MLOps Pipeline with PySpark, GCP, Airflow, Docker, GitLab CI/CD, BigQuery, Power BI, MLflow and pytest**

## 1. Project Overview

This project builds a realistic cloud-native data platform for pharmacovigilance analytics. It processes public adverse-event reports, stores the raw data in a cloud data lake, transforms nested JSON data with PySpark, creates analytics-ready datasets, loads them into BigQuery, and exposes the results through Power BI dashboards.

The project also includes professional software engineering and MLOps practices such as Dockerized environments, Airflow orchestration, pytest-based testing, GitLab CI/CD automation, and MLflow experiment tracking for a seriousness prediction model.

## 2. Main Objective

The objective is to simulate how a pharmaceutical company could process adverse-event data to monitor drug safety and support pharmacovigilance teams.

The platform will:

1. Ingest raw adverse-event reports.
2. Store raw data in a cloud data lake.
3. Clean and transform nested JSON files with PySpark.
4. Create structured silver tables.
5. Build gold analytical tables for business use cases.
6. Load curated datasets into BigQuery.
7. Visualize safety insights in Power BI.
8. Track a machine learning model with MLflow.
9. Test the pipeline with pytest.
10. Automate quality checks with GitLab CI/CD.

## 3. Business Use Case

Pharmacovigilance teams need to monitor adverse-event reports to identify safety trends, frequent reactions, serious cases, and potential drug-reaction signals.

Raw adverse-event data is difficult to analyze directly because it is often:

- nested;
- duplicated;
- semi-structured;
- inconsistent;
- large;
- incomplete;
- difficult to query efficiently.

This project transforms messy pharmacovigilance data into clean, structured, analytics-ready datasets.

The platform should help answer questions such as:

- Which drugs are most frequently reported?
- Which adverse reactions appear most often?
- Which reports are serious?
- Which drug-reaction pairs are most common?
- Which countries report the most cases?
- How do reports evolve over time?
- Which drugs have the highest seriousness rates?
- Can a machine learning model predict whether a case is serious?

## 4. Dataset

The project uses the **openFDA Drug Adverse Event dataset**, based on FAERS data.

The raw data contains adverse-event reports with fields such as:

- safety report ID;
- report date;
- country;
- seriousness flags;
- patient age;
- patient sex;
- drugs involved;
- drug role;
- drug indication;
- adverse reactions;
- outcomes.

The raw data is stored as nested JSON. One adverse-event report can contain multiple drugs, multiple reactions, and patient-level information.

Example simplified raw record:

```json
{
  "safetyreportid": "12345678",
  "receivedate": "20250115",
  "serious": "1",
  "seriousnessdeath": "0",
  "seriousnesshospitalization": "1",
  "primarysourcecountry": "FR",
  "patient": {
    "patientsex": "2",
    "patientonsetage": "67",
    "drug": [
      {
        "medicinalproduct": "HUMIRA",
        "drugcharacterization": "1",
        "drugindication": "RHEUMATOID ARTHRITIS"
      },
      {
        "medicinalproduct": "METHOTREXATE",
        "drugcharacterization": "2"
      }
    ],
    "reaction": [
      {
        "reactionmeddrapt": "PNEUMONIA"
      },
      {
        "reactionmeddrapt": "PYREXIA"
      }
    ]
  }
}
```

The main engineering challenge is to flatten, clean, normalize, aggregate, and serve this nested data in a reliable way.

## 5. Final Architecture

```text
openFDA / FAERS Data
        |
        v
Airflow Ingestion DAG
        |
        v
GCS Bronze Layer
Raw JSON files
        |
        v
PySpark Cleaning Jobs
        |
        v
GCS Silver Layer
Clean Parquet tables
        |
        v
PySpark Aggregation Jobs
        |
        v
GCS Gold Layer
Analytics-ready tables
        |
        v
BigQuery Data Warehouse
        |
        +---------------------+
        |                     |
        v                     v
Power BI Dashboard       ML Training Pipeline
                              |
                              v
                         MLflow Tracking
```

## 6. Tech Stack

| Tool | Role |
| --- | --- |
| Python | Main programming language |
| PySpark | Processing and transforming large nested datasets |
| Google Cloud Storage | Storage for bronze, silver, and gold data layers |
| BigQuery | Analytical data warehouse |
| Airflow | Pipeline orchestration |
| Docker | Reproducible local environment |
| GitLab CI/CD | Automated tests, linting, and Docker builds |
| pytest | Testing ingestion, transformations, data quality, DAGs, and ML code |
| MLflow | Experiment tracking and model metrics |
| Power BI | Final analytics dashboard |
| Terraform | Optional infrastructure as code |
| Ruff / Black | Code quality and formatting |

## 7. Repository Structure

```text
pharma-safety-lakehouse/
│
├── dags/
│   └── pharmacovigilance_pipeline.py
│
├── src/
│   ├── ingestion/
│   │   ├── fetch_openfda_events.py
│   │   └── upload_to_gcs.py
│   │
│   ├── spark_jobs/
│   │   ├── bronze_to_silver_events.py
│   │   ├── build_silver_drugs.py
│   │   ├── build_silver_reactions.py
│   │   ├── build_gold_tables.py
│   │   └── build_ml_features.py
│   │
│   ├── quality/
│   │   ├── schema_checks.py
│   │   └── data_quality_checks.py
│   │
│   ├── ml/
│   │   ├── train_seriousness_model.py
│   │   ├── evaluate_model.py
│   │   └── export_mlflow_metrics.py
│   │
│   └── utils/
│       ├── config.py
│       ├── gcp_helpers.py
│       └── spark_session.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_spark_transformations.py
│   ├── test_data_quality.py
│   ├── test_airflow_dag.py
│   └── test_ml_training.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── powerbi/
│   ├── dashboard.pbix
│   └── screenshots/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_ml_experimentation.ipynb
│
├── terraform/
│   ├── main.tf
│   ├── storage.tf
│   ├── bigquery.tf
│   └── variables.tf
│
├── docs/
│   ├── architecture.md
│   ├── data_model.md
│   ├── pipeline_steps.md
│   └── screenshots/
│
├── .gitlab-ci.yml
├── requirements.txt
├── pyproject.toml
├── Makefile
├── README.md
└── .env.example
```

## 8. Data Lakehouse Design

The project uses a **bronze / silver / gold** data architecture.

### 8.1 Bronze Layer

The bronze layer stores the raw JSON files exactly as downloaded.

Example path:

```text
gcs://pharma-safety-lakehouse/bronze/openfda_events/year=2025/month=01/events.json
```

Main actions:

- download adverse-event data;
- save raw JSON files;
- partition files by year and month;
- keep the original raw content unchanged.

Final product:

```text
Raw adverse-event JSON files stored in Google Cloud Storage
```

Purpose:

- traceability;
- reprocessing;
- raw backup;
- auditability.

### 8.2 Silver Layer

The silver layer contains cleaned and normalized tables created from the raw nested JSON data.

Example output paths:

```text
gcs://pharma-safety-lakehouse/silver/adverse_events/
gcs://pharma-safety-lakehouse/silver/drugs/
gcs://pharma-safety-lakehouse/silver/reactions/
gcs://pharma-safety-lakehouse/silver/patients/
```

Main transformations:

- parse report dates;
- convert numeric flags;
- normalize patient sex;
- clean drug names;
- clean reaction terms;
- remove duplicate reports;
- explode nested drug arrays;
- explode nested reaction arrays;
- create clean relational tables;
- write outputs as Parquet files.

Final products:

```text
silver_adverse_events
silver_drugs
silver_reactions
silver_patients
```

Example `silver_adverse_events` table:

| safety_report_id | report_date | country | is_serious | death | hospitalization | patient_age | patient_sex |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 12345678 | 2025-01-15 | FR | 1 | 0 | 1 | 67 | Female |

Example `silver_drugs` table:

| safety_report_id | drug_name | drug_role | indication |
| --- | --- | --- | --- |
| 12345678 | HUMIRA | Suspect | Rheumatoid Arthritis |
| 12345678 | METHOTREXATE | Concomitant | null |

Example `silver_reactions` table:

| safety_report_id | reaction_name |
| --- | --- |
| 12345678 | Pneumonia |
| 12345678 | Pyrexia |

### 8.3 Gold Layer

The gold layer contains business-ready datasets designed for analytics, dashboards, and machine learning.

Example output paths:

```text
gcs://pharma-safety-lakehouse/gold/safety_overview/
gcs://pharma-safety-lakehouse/gold/drug_summary/
gcs://pharma-safety-lakehouse/gold/drug_reaction_pairs/
gcs://pharma-safety-lakehouse/gold/ml_features/
```

Final products:

```text
gold_safety_overview
gold_monthly_trends
gold_drug_summary
gold_reaction_summary
gold_drug_reaction_pairs
gold_signal_detection
gold_ml_features
```

Example `gold_drug_summary` table:

| drug_name | total_reports | serious_reports | seriousness_rate | top_reaction |
| --- | ---: | ---: | ---: | --- |
| HUMIRA | 15,200 | 5,700 | 37.5% | Pneumonia |
| IBUPROFEN | 9,300 | 1,200 | 12.9% | Vomiting |

Example `gold_drug_reaction_pairs` table:

| drug_name | reaction_name | pair_count | serious_count | seriousness_rate |
| --- | --- | ---: | ---: | ---: |
| HUMIRA | Pneumonia | 2,400 | 1,200 | 50.0% |
| IBUPROFEN | Vomiting | 1,100 | 90 | 8.2% |

## 9. Detailed Project Phases

### Phase 1 — Project Setup

Goal: create a clean, professional repository.

Tasks:

- create a GitLab repository;
- create the folder structure;
- add an initial README;
- add `.gitignore`;
- add `requirements.txt`;
- add `pyproject.toml`;
- add a `Makefile`;
- add `.env.example`;
- set up the local Python environment.

Deliverables:

- clean repository;
- initial README;
- basic Python package structure.

Recommended README sections:

- project overview;
- architecture diagram;
- tech stack;
- dataset;
- local setup;
- pipeline steps;
- dashboard preview;
- CI/CD explanation.

### Phase 2 — Local Docker Environment

Goal: make the project reproducible locally.

Create Docker services for:

- Airflow webserver;
- Airflow scheduler;
- PostgreSQL metadata database;
- Spark master;
- Spark worker;
- MLflow tracking server;
- Jupyter notebook.

Simplified Docker Compose architecture:

```text
docker-compose
│
├── airflow-webserver
├── airflow-scheduler
├── postgres
├── spark-master
├── spark-worker
├── mlflow
└── jupyter
```

Deliverables:

- `Dockerfile`;
- `docker-compose.yml`;
- working Airflow UI;
- working Spark environment;
- working MLflow UI.

Useful commands:

```bash
make up
make down
make test
make format
make lint
```

### Phase 3 — Raw Data Ingestion

Goal: download raw adverse-event data.

Script:

```text
src/ingestion/fetch_openfda_events.py
```

Responsibilities:

- call the openFDA API or load downloaded JSON files;
- filter data by year and month;
- save raw JSON locally;
- upload raw JSON to the GCS bronze bucket;
- log the number of records;
- handle errors and retries.

Local output example:

```text
data/bronze/openfda_events/year=2025/month=01/events.json
```

Cloud output example:

```text
gcs://pharma-safety-lakehouse/bronze/openfda_events/year=2025/month=01/events.json
```

Deliverables:

- working ingestion script;
- raw files saved locally;
- raw files uploaded to GCS;
- basic ingestion logs.

MVP recommendation:

Start with one month of data only before scaling the pipeline to larger volumes.

### Phase 4 — Bronze to Silver Transformation with PySpark

Goal: transform raw nested JSON into clean structured tables.

Script:

```text
src/spark_jobs/bronze_to_silver_events.py
```

Input:

```text
bronze/openfda_events/year=2025/month=01/events.json
```

Outputs:

```text
silver/adverse_events/
silver/drugs/
silver/reactions/
silver/patients/
```

Main transformations:

- read raw JSON with Spark;
- select useful fields;
- flatten patient structure;
- explode the `patient.drug` array;
- explode the `patient.reaction` array;
- parse `receivedate`;
- convert seriousness flags;
- normalize patient sex;
- clean drug names;
- clean reaction names;
- remove duplicate `safety_report_id` values;
- write Parquet tables.

Example transformations:

```text
receivedate = "20250115"
        ↓
report_date = 2025-01-15
```

```text
patientsex = "2"
        ↓
patient_sex = "Female"
```

```text
serious = "1"
        ↓
is_serious = 1
```

Deliverables:

- `silver_adverse_events` Parquet table;
- `silver_drugs` Parquet table;
- `silver_reactions` Parquet table;
- `silver_patients` Parquet table.

### Phase 5 — Data Quality Checks

Goal: improve pipeline reliability and data trust.

Script:

```text
src/quality/data_quality_checks.py
```

Checks:

- `safety_report_id` must not be null;
- `report_date` must be valid;
- `is_serious` must be 0 or 1;
- `patient_age` must be realistic;
- `drug_name` should not be empty in the drug table;
- `reaction_name` should not be empty in the reaction table;
- duplicate rows should remain below an acceptable threshold;
- row count should be above a minimum threshold;
- required columns should exist.

Example checks:

```python
assert df.filter("safety_report_id IS NULL").count() == 0
assert df.select("safety_report_id").distinct().count() == df.count()
```

Deliverables:

- data quality validation script;
- Airflow task for data quality checks;
- pytest tests for quality rules.

### Phase 6 — Build Gold Analytical Tables

Goal: create business-ready datasets.

Script:

```text
src/spark_jobs/build_gold_tables.py
```

Inputs:

```text
silver_adverse_events
silver_drugs
silver_reactions
```

Outputs:

```text
gold_safety_overview
gold_monthly_trends
gold_drug_summary
gold_reaction_summary
gold_drug_reaction_pairs
gold_signal_detection
gold_ml_features
```

#### `gold_safety_overview`

Contains global KPIs:

- total reports;
- serious reports;
- death reports;
- hospitalization reports;
- unique drugs;
- unique reactions;
- seriousness rate.

#### `gold_monthly_trends`

Contains time-based metrics:

- month;
- total reports;
- serious reports;
- death reports;
- hospitalization reports;
- seriousness rate.

#### `gold_drug_summary`

Contains drug-level metrics:

- drug name;
- total reports;
- serious reports;
- seriousness rate;
- top reaction.

#### `gold_drug_reaction_pairs`

Contains pair-level metrics:

- drug name;
- reaction name;
- pair count;
- serious count;
- seriousness rate.

#### `gold_ml_features`

Contains one row per report for machine learning:

- safety report ID;
- patient age;
- patient sex;
- country;
- number of drugs;
- number of reactions;
- suspect drug indicator;
- seriousness label.

Deliverables:

- gold Parquet tables;
- gold BigQuery tables;
- documented data model.

### Phase 7 — Load Gold Tables into BigQuery

Goal: make the data available for BI and analytics.

BigQuery datasets:

```text
pharma_safety_silver
pharma_safety_gold
```

BigQuery tables:

```text
pharma_safety_gold.safety_overview
pharma_safety_gold.monthly_trends
pharma_safety_gold.drug_summary
pharma_safety_gold.reaction_summary
pharma_safety_gold.drug_reaction_pairs
pharma_safety_gold.ml_features
```

Example SQL query:

```sql
SELECT
  drug_name,
  total_reports,
  serious_reports,
  seriousness_rate
FROM pharma_safety_gold.drug_summary
ORDER BY serious_reports DESC
LIMIT 20;
```

Deliverables:

- BigQuery dataset created;
- gold tables loaded;
- example SQL queries added to the README.

### Phase 8 — Airflow Orchestration

Goal: orchestrate the complete pipeline.

DAG file:

```text
dags/pharmacovigilance_pipeline.py
```

DAG flow:

```text
start
  |
  v
ingest_openfda_data
  |
  v
upload_raw_to_gcs
  |
  v
bronze_to_silver_spark_job
  |
  v
run_data_quality_checks
  |
  v
build_gold_tables
  |
  v
load_gold_to_bigquery
  |
  v
train_seriousness_model
  |
  v
export_ml_metrics
  |
  v
end
```

Airflow features to include:

- task retries;
- retry delay;
- failure handling;
- logging;
- scheduled interval;
- task dependencies;
- environment variables;
- DAG validation.

Recommended DAG schedule:

```text
Monthly
```

A monthly batch schedule is realistic for this type of pharmacovigilance analytics pipeline.

Deliverables:

- working Airflow DAG;
- screenshot of a successful DAG run;
- DAG explanation in the README.

### Phase 9 — MLflow Model Tracking

Goal: add a small MLOps component to the data platform.

ML task:

```text
Predict whether an adverse-event report is serious.
```

Target variable:

```text
is_serious
```

Candidate features:

- patient age;
- patient sex;
- country;
- number of drugs;
- number of reactions;
- suspect drug indicator;
- report year.

Models:

- Logistic Regression;
- Random Forest;
- XGBoost or LightGBM.

Recommended first implementation:

1. Logistic Regression.
2. Random Forest.

Metrics and artifacts to track in MLflow:

- model name;
- parameters;
- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- confusion matrix;
- model artifact;
- training date;
- dataset version.

Script:

```text
src/ml/train_seriousness_model.py
```

Deliverables:

- MLflow experiment;
- tracked metrics;
- saved model artifact;
- screenshot of MLflow UI;
- optional exported metrics table for Power BI.

The machine learning component supports the platform, while the main focus remains the data engineering pipeline.

### Phase 10 — Power BI Dashboard

Goal: create a final business-facing dashboard connected to BigQuery.

#### Page 1 — Safety Overview

Visuals:

- total reports card;
- serious reports card;
- death reports card;
- hospitalization reports card;
- reports by month line chart;
- reports by country map or bar chart;
- patient sex distribution;
- age group distribution.

Data sources:

```text
gold_safety_overview
gold_monthly_trends
```

#### Page 2 — Drug Analysis

Visuals:

- top reported drugs;
- top serious drugs;
- seriousness rate by drug;
- selected drug trend;
- top reactions for selected drug.

Data sources:

```text
gold_drug_summary
gold_drug_reaction_pairs
```

#### Page 3 — Reaction and Signal Detection

Visuals:

- top adverse reactions;
- top drug-reaction pairs;
- serious drug-reaction pairs;
- potential signal ranking;
- pair count versus seriousness rate.

Data source:

```text
gold_drug_reaction_pairs
```

#### Page 4 — ML Monitoring

Visuals:

- latest model version;
- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- prediction distribution;
- feature importance.

Data sources:

```text
MLflow exported metrics
gold_ml_features
```

Deliverables:

- Power BI `.pbix` file;
- dashboard screenshots;
- dashboard explanation in the README.

### Phase 11 — Testing with pytest

Goal: demonstrate reliable software engineering practices.

Test files:

```text
tests/test_ingestion.py
tests/test_spark_transformations.py
tests/test_data_quality.py
tests/test_airflow_dag.py
tests/test_ml_training.py
```

Test examples:

- ingestion returns records;
- raw file is saved;
- required columns exist;
- date parsing works;
- patient sex mapping works;
- serious flag conversion works;
- drug array explosion works;
- reaction array explosion works;
- duplicate IDs are removed;
- Airflow DAG imports successfully;
- ML training produces metrics.

Example transformation test:

```python
def test_serious_flag_is_binary(clean_events_df):
    values = [
        row["is_serious"]
        for row in clean_events_df.select("is_serious").distinct().collect()
    ]
    assert set(values).issubset({0, 1})
```

Deliverables:

- pytest test suite;
- CI/CD pipeline running tests automatically;
- screenshot or badge showing passing tests.

### Phase 12 — GitLab CI/CD

Goal: automate code quality checks and build validation.

File:

```text
.gitlab-ci.yml
```

Pipeline stages:

```text
lint
test
build
validate
```

Recommended jobs:

- run Ruff;
- run Black in check mode;
- run pytest;
- validate Airflow DAG imports;
- build Docker image;
- optionally deploy DAGs to GCP.

Example pipeline:

```yaml
stages:
  - lint
  - test
  - build

lint:
  stage: lint
  script:
    - ruff check src tests dags
    - black --check src tests dags

test:
  stage: test
  script:
    - pytest tests/

build:
  stage: build
  script:
    - docker build -t pharma-safety-lakehouse .
```

Deliverables:

- working GitLab CI/CD pipeline;
- screenshot of a successful pipeline;
- CI/CD explanation in the README.

## 10. MVP Scope

The first version should focus on processing one month of adverse-event data from raw JSON to dashboard.

MVP includes:

- one month of openFDA data;
- bronze raw storage;
- PySpark silver transformation;
- three gold tables;
- BigQuery loading;
- basic Airflow DAG;
- basic Power BI dashboard;
- basic pytest tests.

MVP gold tables:

```text
gold_safety_overview
gold_drug_summary
gold_drug_reaction_pairs
```

MVP dashboard pages:

```text
Safety Overview
Drug Analysis
```

## 11. Advanced Scope

After the MVP is complete, the project can be extended with more advanced features.

Advanced features:

- multiple months of data;
- incremental ingestion;
- more data quality checks;
- MLflow tracking;
- seriousness prediction model;
- Power BI ML monitoring page;
- GitLab CI/CD Docker build;
- Terraform infrastructure;
- signal detection metrics;
- improved README screenshots.

Optional pharmacovigilance metric:

```text
Reporting Odds Ratio, also called ROR
```

## 12. Suggested Build Order

Recommended implementation order:

1. Create the repository and folder structure.
2. Download a small sample of raw openFDA data.
3. Explore the JSON structure in a notebook.
4. Write a PySpark job to flatten one raw file.
5. Create `silver_adverse_events`.
6. Create `silver_drugs`.
7. Create `silver_reactions`.
8. Build `gold_safety_overview`.
9. Build `gold_drug_summary`.
10. Build `gold_drug_reaction_pairs`.
11. Load gold tables to BigQuery.
12. Create a simple Power BI dashboard.
13. Create the Airflow DAG locally.
14. Dockerize the environment.
15. Add pytest tests.
16. Add GitLab CI/CD.
17. Add the ML training script.
18. Track ML experiments with MLflow.
19. Improve the dashboard.
20. Polish the README with screenshots.

This order keeps the project focused on the data first, before adding infrastructure and automation.

## 13. Timeline

### Week 1 — Data and Local Pipeline

- understand the dataset;
- download sample data;
- explore the JSON structure;
- build the PySpark flattening job;
- create silver tables locally.

### Week 2 — Gold Tables and BigQuery

- create gold tables;
- write aggregation logic;
- load outputs into BigQuery;
- write SQL examples;
- start the README data model section.

### Week 3 — Airflow and Docker

- create the Airflow DAG;
- run tasks locally;
- add Docker Compose;
- add logs and retries;
- add screenshots.

### Week 4 — Tests and CI/CD

- add pytest tests;
- add linting;
- create the GitLab CI/CD pipeline;
- build Docker image in CI;
- validate DAGs.

### Week 5 — Power BI and MLflow

- connect Power BI to BigQuery;
- build dashboard pages;
- train the seriousness model;
- track experiments with MLflow;
- export metrics.

### Week 6 — Final Polish

- improve the README;
- add an architecture diagram;
- add screenshots;
- add project explanation;
- add interview pitch;
- clean the repository;
- optionally record a short demo.

## 14. Final Deliverables

At the end of the project, the repository should contain:

- working PySpark pipeline;
- bronze, silver, and gold lakehouse structure;
- Airflow DAG;
- BigQuery tables;
- Power BI dashboard;
- MLflow experiment tracking;
- pytest test suite;
- GitLab CI/CD pipeline;
- Docker environment;
- professional README;
- architecture diagram;
- screenshots;
- optional Terraform configuration.

The README should include:

- project overview;
- business problem;
- architecture diagram;
- dataset description;
- data model;
- pipeline steps;
- local execution instructions;
- GCP setup;
- Airflow DAG screenshot;
- BigQuery tables;
- Power BI screenshots;
- MLflow screenshot;
- testing strategy;
- CI/CD pipeline screenshot;
- future improvements.

## 15. Interview Pitch

The raw data starts as nested adverse-event JSON from openFDA. It is stored unchanged in a bronze layer on Google Cloud Storage. PySpark is then used to clean and flatten the raw data into silver tables such as adverse events, drugs, reactions, and patients. After that, gold analytical tables are created for safety overview, monthly trends, drug summaries, and drug-reaction pairs. These gold tables are loaded into BigQuery and visualized in Power BI. The full pipeline is orchestrated with Airflow, tested with pytest, containerized with Docker, and validated through GitLab CI/CD. MLflow is added to track a model that predicts whether an adverse-event report is serious.

## 16. Tool Roles

Each tool in the project has a clear purpose:

| Tool | Purpose |
| --- | --- |
| PySpark | Process complex nested data |
| GCP / GCS | Store data lake layers |
| BigQuery | Serve analytics-ready tables |
| Airflow | Orchestrate the pipeline |
| Docker | Reproduce the environment |
| GitLab CI/CD | Automate quality checks |
| pytest | Validate reliability |
| Power BI | Deliver business insights |
| MLflow | Track ML experiments |

The project is designed as a realistic data platform where every technology supports a specific part of the system.
