# 📦 Stockwise — Stockout Intelligence

**Predict inventory stockout risk. Understand inventory pressure. Make smarter stocking decisions.**

Stockwise is a machine-learning-powered inventory analytics dashboard designed to help identify products that may be at risk of running out of stock. It combines historical sales information, inventory indicators, engineered features, and a custom Logistic Regression model to analyze retail inventory data.

The project integrates a Python backend with an interactive web dashboard, allowing users to upload CSV files, analyze inventory records, visualize stock conditions, and export analyzed results.

---

## ✨ Key Features

- **📂 CSV Upload:** Upload retail inventory datasets for analysis.
- **🤖 Custom Machine Learning:** Use a scratch-built Logistic Regression pipeline to estimate stockout risk.
- **📊 Interactive Dashboard:** View inventory KPIs, risk distributions, and sales trends.
- **🔍 Inventory Risk Analysis:** Identify healthy, at-risk, and critical inventory conditions using operational indicators.
- **🧠 Feature Engineering:** Derive rolling sales, sales trends, inventory-to-demand ratios, and stock pressure indicators.
- **📥 Export Results:** Download analyzed inventory data as a CSV file.
- **🧹 Data Validation:** Check required columns, convert numeric values, parse dates, and prepare data for analysis.
- **🌐 Deployment Ready:** Structure the application for cloud deployment, subject to platform and resource limitations.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Flask | Backend and API endpoints |
| NumPy | Numerical computation and model calculations |
| Pandas | CSV processing and feature engineering |
| HTML5 | Dashboard structure |
| CSS3 | Styling and responsive layout |
| JavaScript | Dashboard interactions and API communication |
| Matplotlib / Chart library | Use only if included in your implementation |
| GitHub | Version control and project hosting |
| Vercel / Render | Deployment options |

---

## 🧠 Machine Learning Approach

Stockwise uses a **custom Logistic Regression implementation** rather than relying entirely on a prebuilt classification estimator.

The trained model and its scaler are stored in `.npz` files and loaded by the application.

### Model workflow

1. Upload the inventory dataset.
2. Validate the required columns.
3. Parse dates and numeric values.
4. Sort records by store, product, and date.
5. Calculate historical sales features.
6. Apply the saved feature scaler.
7. Generate stockout-risk probabilities.
8. Assign a risk classification using a configurable decision threshold.
9. Display operational inventory metrics and export the analyzed records.

### Feature engineering

The model uses 12 features in a fixed order:

| Feature | Description |
|---|---|
| `Inventory Level` | Current recorded inventory |
| `Units Sold` | Units sold in the recorded period |
| `Units Ordered` | Units ordered |
| `Demand Forecast` | Forecast demand |
| `rolling_7d_sales` | Historical seven-period rolling sales |
| `rolling_14d_sales` | Historical fourteen-period rolling sales |
| `inventory_to_demand_ratio` | Inventory relative to estimated daily sales |
| `sales_trend` | Difference between recent and previous rolling sales |
| `stock_pressure_ratio` | Sales relative to available inventory and incoming orders |
| `Price` | Product price |
| `Discount` | Applied discount |
| `Competitor Pricing` | Competitor's recorded price |

**Important:** Rolling sales and trend features depend on historical records for each store-product combination. The first records may be excluded when sufficient history is unavailable.

### Risk prediction

The model generates a probability estimate and assigns a classification using a threshold.

For example, with a threshold of `0.40`:

- Probability below `0.40` → **Low Risk**
- Probability at or above `0.40` → **At Risk**

This threshold is a configuration choice, not a guarantee that the model is calibrated or accurate at that probability.

### Operational status vs. ML risk

Stockwise distinguishes between two concepts:

- **Operational stock status:** Evaluates inventory against demand-related indicators to classify inventory as healthy, at risk, or critical.
- **ML stockout risk:** Uses the trained Logistic Regression model to estimate the likelihood of the target event represented in its training data.

These classifications are not interchangeable. A product can have an operational warning without receiving the same machine-learning classification.

---

## 📁 Project Structure

```text
Stockwise/
│
├── app.py
├── logisticRegrassionPipeline.py
├── logistic_model.npz
├── scaler.npz
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    ├── styles.css
    └── script.js
```

**File responsibilities**

- `app.py` — Flask routes, data preparation, predictions, dashboard metrics, and CSV export.
- `logisticRegrassionPipeline.py` — Custom Logistic Regression implementation and model utilities.
- `logistic_model.npz` — Saved model parameters.
- `scaler.npz` — Saved scaling parameters.
- `templates/index.html` — Dashboard interface.
- `static/styles.css` — Dashboard styles.
- `static/script.js` — API requests, charts, table rendering, uploads, and export interactions.
- `requirements.txt` — Python dependencies.

The model and scaler filenames above assume they match the files used by your actual implementation.

---

## ⚙️ Installation and Local Setup

### Prerequisites

- Python 3.10 or a compatible version supported by your dependencies
- pip
- Git (optional)

### 1. Clone the repository

```bash
git clone https://github.com/arpit4baj/Stockout_model.git
cd Stockout_model

```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Example `requirements.txt`:

```text
Flask
numpy
pandas
gunicorn
```

Add any additional third-party packages imported by your implementation.

### 4. Run the application

```bash
python app.py
```

Open the local URL printed by Flask, typically:

```text
http://127.0.0.1:5000
```

Upload a CSV file that contains the required columns and enough historical records for feature engineering.

---

## 📄 Dataset Requirements

The input CSV is expected to contain the following base columns:

```text
Date
Store ID
Product ID
Inventory Level
Units Sold
Units Ordered
Demand Forecast
Price
Discount
Competitor Pricing
```

The remaining engineered features are calculated by the application.

### Data preparation notes

- Dates must be parseable.
- Numeric fields should contain valid numeric values.
- Records should represent the correct store-product history.
- Historical observations must be available to calculate rolling features.
- The input schema and units should match the model's training data.

The application may discard records that do not have enough historical information to calculate the required features. Therefore, the number of analyzed rows can be lower than the number uploaded.

---

## 🚀 Deployment

### Option A — Vercel

Vercel can host supported Python applications and is useful for a portfolio demonstration.

General steps:

1. Push the project to GitHub.
2. Import the repository through [Vercel](https://vercel.com/).
3. Configure the Python/Flask application according to the current Vercel documentation.
4. Verify that the model and scaler files are included in the deployment.
5. Test the upload, prediction, dashboard, and export workflows.

Official guide: [Deploy Flask on Vercel](https://vercel.com/kb/guide/ship-a-flask-app-on-vercel)

**Vercel limitations to consider**

- Vercel Functions have a documented 4.5 MB request and response body limit; a large CSV sent directly to a Flask upload endpoint may exceed it.
- Function execution time and memory are limited by the selected deployment plan.
- Local function storage should not be treated as permanent storage.
- Large datasets may require direct-to-object-storage uploads or a separate backend.
- Model loading, feature engineering, and predictions must complete within the function's resource limits.

### Option B — Render

Render is another option for hosting the existing Flask backend.

1. Push the repository to GitHub.
2. Create a new Web Service in [Render](https://render.com/).
3. Set the build command to:

   ```bash
   pip install -r requirements.txt
   ```

4. Set the start command to:

   ```bash
   gunicorn app:app
   ```

5. Deploy and test the application using the provided URL.

Official guide: [Deploy Flask on Render](https://render.com/docs/deploy-flask)

Choose a hosting plan based on dataset size, memory use, execution time, and traffic. Neither platform automatically guarantees that large datasets will process successfully.

---

## ⚠️ Current Limitations

Stockwise is a machine-learning inventory analysis project, not yet a guaranteed production-grade stockout prevention system.

### 1. Historical data dependency

Rolling sales, sales trends, and other engineered features require sufficient historical data. Records without the required history may be removed from analysis.

### 2. Model performance has not been independently established here

No verified test-set metrics are provided in this README. Accuracy, precision, recall, F1-score, ROC-AUC, and calibration should be measured on data that was not used to train the model before making performance claims.

### 3. Probability estimates are not guarantees

A predicted probability does not mean that a stockout will definitely occur. The model's threshold should be selected using validation data and the relative cost of false alarms versus missed stockouts.

### 4. Training and inference must remain consistent

The model depends on the same feature definitions, feature order, scaling parameters, and target meaning used during training. Changing feature engineering or supplying incompatible `.npz` files can produce incorrect results or runtime errors.

### 5. Dataset quality affects results

Missing values, inconsistent dates, duplicate records, inaccurate demand forecasts, and inconsistent product identifiers can affect feature calculations and predictions. Input validation cannot guarantee that the underlying data is correct.

### 6. CSV upload and deployment resource limits

Large CSV files can consume significant memory and processing time. Serverless platforms impose request-size, memory, and execution-time limits. Larger workloads may need chunked processing, background jobs, or a dedicated backend.

### 7. Temporary storage and persistence

In-memory data and local files in serverless environments should not be assumed to survive restarts or redeployments. Persistent datasets, user uploads, and historical analyses require an appropriate database or object-storage service.

### 8. Operational status is a heuristic

Healthy, at-risk, and critical classifications depend on the operational rules implemented in the application. They should not be interpreted as independently validated predictions unless a separate evaluation supports that claim.

### 9. No automatic inventory replenishment

Stockwise analyzes inventory records; it does not automatically place orders, guarantee supply availability, or account for every real-world factor such as lead times, supplier delays, perishability, and sudden demand changes.

---

## 🔮 Future Improvements

- Add a model evaluation report with precision, recall, F1-score, and ROC-AUC.
- Validate the model using time-based train, validation, and test splits.
- Calibrate predicted probabilities and optimize the decision threshold.
- Add configurable stockout warning thresholds.
- Introduce supplier lead times and reorder-point recommendations.
- Add database-backed storage for historical analyses.
- Support larger files through asynchronous processing or cloud object storage.
- Add automated data-quality reports and duplicate detection.
- Monitor prediction drift and retrain the model when appropriate.
- Add automated tests for the API, feature engineering, and model loading.

---

## 🔐 Responsible Use

Use Stockwise as a decision-support tool. Review critical warnings alongside actual inventory, upcoming deliveries, supplier lead times, and business context. Do not rely on the model alone for purchasing or replenishment decisions.

Before sharing a deployment, remove sensitive datasets and credentials from the repository. Avoid exposing customer, supplier, or commercially confidential inventory information.

---

## 👨‍💻 Project Goals

Stockwise demonstrates how a custom machine-learning pipeline can be integrated into a practical inventory analytics application. It combines feature engineering, model inference, backend development, data visualization, and deployment considerations in a single project.

**Built to explore the intersection of machine learning, retail analytics, and practical decision support.**

---

*Note: Update the README whenever the model, dataset schema, deployment configuration, or measured evaluation results change. Add actual metrics only after testing the saved model on an appropriate held-out dataset.*
