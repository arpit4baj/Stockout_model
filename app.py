
from flask import Flask, render_template, request, jsonify, send_file
from logistic_model import LogisticRegScratch

import pandas as pd
import numpy as np
import io
import os
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# These features and their order must match model training.
FEATURES = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "rolling_7d_sales",
    "rolling_14d_sales",
    "inventory_to_demand_ratio",
    "sales_trend",
    "stock_pressure_ratio",
    "Price",
    "Discount",
    "Competitor Pricing",
]

IDENTIFIER_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
]

BASE_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Competitor Pricing",
]

# No demo data. The dashboard remains empty until upload.
df = pd.DataFrame()

MODEL = None
SCALER_MEAN = None
SCALER_STD = None

PREDICTION_THRESHOLD = 0.40


# =========================================================
# LOAD TRAINED MODEL AND SCALER
# =========================================================

def load_trained_model():
    model_path = BASE_DIR / "logistic_model.npz"
    scaler_path = BASE_DIR / "scaler.npz"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}"
        )

    model = LogisticRegScratch()
    model.load_model(str(model_path))

    with np.load(scaler_path) as scaler:
        mean = np.asarray(scaler["mean"], dtype=float)
        std = np.asarray(scaler["std"], dtype=float)

    if len(mean) != len(FEATURES):
        raise ValueError(
            "Scaler feature count does not match FEATURES."
        )

    if len(std) != len(FEATURES):
        raise ValueError(
            "Scaler standard deviation count does not match FEATURES."
        )

    if len(model.weights) != len(FEATURES):
        raise ValueError(
            "Model feature count does not match FEATURES."
        )

    if not np.isfinite(mean).all():
        raise ValueError("Scaler mean contains invalid values.")

    if not np.isfinite(std).all():
        raise ValueError("Scaler standard deviation contains invalid values.")

    # Prevent division by zero for constant training features.
    std = np.where(std == 0, 1.0, std)

    return model, mean, std


try:
    MODEL, SCALER_MEAN, SCALER_STD = load_trained_model()
except Exception as error:
    print(f"MODEL LOADING ERROR: {error}")
    # Flask can start, but predictions remain unavailable
    # until the model and scaler files are corrected.


# =========================================================
# PREPARE UPLOADED DATA AND ENGINEER FEATURES
# =========================================================

def prepare_data(data):
    """
    Validate uploaded CSV data and calculate model features
    from historical sales for each Store ID / Product ID pair.
    """

    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()

    # Handle accidental spaces in identifier values.
    for column in ["Store ID", "Product ID"]:
        if column in data.columns:
            data[column] = data[column].astype("string").str.strip()

    missing = [
        column for column in BASE_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing)
        )

    # Convert date column.
    data["Date"] = pd.to_datetime(
        data["Date"], errors="coerce"
    )

    # Convert original numeric columns.
    numeric_columns = [
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Competitor Pricing",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(
        subset=BASE_COLUMNS
    ).copy()

    if data.empty:
        raise ValueError(
            "No valid rows remain after validating the CSV."
        )

    # Preserve the original order so the output can be restored
    # to the same sequence after feature calculations.
    data["_original_order"] = np.arange(len(data))

    # Rolling calculations require chronological ordering.
    data = data.sort_values(
        ["Store ID", "Product ID", "Date", "_original_order"],
        kind="stable"
    ).reset_index(drop=True)

    grouped_sales = data.groupby(
        ["Store ID", "Product ID"],
        sort=False
    )["Units Sold"]

    # Sales in the previous 7 records, excluding current record.
    data["rolling_7d_sales"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(
            window=7, min_periods=7
        ).sum()
    )

    # Sales in the previous 14 records, excluding current record.
    data["rolling_14d_sales"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(
            window=14, min_periods=14
        ).sum()
    )

    # Sales in the 7 records before the most recent 7 records.
    data["previous_7d_sales"] = grouped_sales.transform(
        lambda series: series.shift(8).rolling(
            window=7, min_periods=7
        ).sum()
    )

    # Inventory relative to average daily sales.
    average_daily_sales = data["rolling_7d_sales"] / 7.0

    data["inventory_to_demand_ratio"] = (
        data["Inventory Level"]
        / average_daily_sales.replace(0, np.nan)
    )

    # Compare recent weekly sales with the preceding week.
    data["sales_trend"] = (
        data["rolling_7d_sales"]
        - data["previous_7d_sales"]
    )

    # Current sales relative to inventory plus incoming orders.
    available_stock = (
        data["Inventory Level"] + data["Units Ordered"]
    )

    data["stock_pressure_ratio"] = (
        data["Units Sold"]
        / available_stock.replace(0, np.nan)
    )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    # Rows without sufficient sales history or defined ratios
    # cannot be passed to the trained model.
    data = data.dropna(
        subset=FEATURES
    ).copy()

    if data.empty:
        raise ValueError(
            "Not enough historical sales data to calculate model "
            "features. Each Store ID / Product ID group needs at "
            "least 21 ordered records with valid sales and non-zero "
            "denominators for all calculated features."
        )

    data = data.sort_values(
        "_original_order", kind="stable"
    )

    data = data.drop(
        columns=["_original_order", "previous_7d_sales"],
        errors="ignore"
    )

    return data.reset_index(drop=True)


# =========================================================
# MACHINE LEARNING PREDICTIONS
# =========================================================

def predict_stockout_risk(data):
    """
    Apply the saved scaler and trained logistic regression model.
    """

    if MODEL is None or SCALER_MEAN is None or SCALER_STD is None:
        raise RuntimeError(
            "The trained model or scaler could not be loaded. "
            "Check logistic_model.npz and scaler.npz."
        )

    result = data.copy()

    # Keep the exact feature order used during training.
    X = result[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "Model features contain missing or infinite values."
        )

    X_scaled = (X - SCALER_MEAN) / SCALER_STD

    probabilities = np.asarray(
        MODEL.predict_proba(X_scaled)
    )

    # Support implementations returning either (n,) / (n, 1)
    # or two class probabilities in an (n, 2) array.
    if probabilities.ndim == 2:
        if probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]
        elif probabilities.shape[1] == 1:
            probabilities = probabilities[:, 0]
        else:
            raise ValueError(
                "Unexpected shape returned by predict_proba()."
            )

    probabilities = probabilities.reshape(-1)

    if len(probabilities) != len(result):
        raise ValueError(
            "Prediction count does not match uploaded row count."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "The model returned invalid prediction probabilities."
        )

    probabilities = np.clip(probabilities, 0, 1)

    result["Stockout Risk Probability"] = np.round(
        probabilities * 100, 2
    )

    result["Stockout Risk Prediction"] = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(int)

    result["ML Risk Status"] = np.where(
        result["Stockout Risk Prediction"] == 1,
        "At Risk",
        "Low Risk"
    )

    return result


# =========================================================
# OPERATIONAL INVENTORY STATUS
# =========================================================

def add_operational_status(data):
    result = data.copy()

    inventory = result["Inventory Level"].clip(lower=0)
    demand = result["Demand Forecast"].clip(lower=0)

    result["Stock Status"] = np.select(
        [
            (demand > 0) & (inventory <= demand * 0.5),
            (demand > 0) & (inventory <= demand),
        ],
        [
            "Critical",
            "At Risk",
        ],
        default="Healthy"
    )

    result["Estimated Coverage"] = np.where(
        demand > 0,
        np.round(inventory / demand, 2),
        np.nan
    )

    return result


# =========================================================
# DASHBOARD METRICS
# =========================================================

def get_dashboard_metrics(data):
    inventory = data["Inventory Level"].clip(lower=0)
    demand = data["Demand Forecast"].clip(lower=0)

    operational_at_risk = (
        (inventory <= demand) & (demand > 0)
    )

    operational_critical = (
        (inventory <= demand * 0.5) & (demand > 0)
    )

    ml_at_risk = (
        data["Stockout Risk Prediction"] == 1
    )

    return {
        "total_products": int(len(data)),
        "total_records": int(len(data)),
        "at_risk": int(operational_at_risk.sum()),
        "critical": int(operational_critical.sum()),
        "total_inventory": float(inventory.sum()),
        "total_units_sold": float(
            data["Units Sold"].clip(lower=0).sum()
        ),
        "average_demand": round(
            float(demand.mean()), 2
        ),
        "risk_percentage": round(
            float(operational_at_risk.mean() * 100), 1
        ),
        "ml_at_risk": int(ml_at_risk.sum()),
        "ml_risk_percentage": round(
            float(ml_at_risk.mean() * 100), 1
        ),
    }


# =========================================================
# INVENTORY TABLE
# =========================================================

def get_inventory_table(data):
    table = add_operational_status(data)

    display_columns = [
        column for column in IDENTIFIER_COLUMNS
        if column in table.columns
    ]

    display_columns += [
        "Inventory Level",
        "Units Sold",
        "Demand Forecast",
        "Stock Status",
        "Estimated Coverage",
        "Stockout Risk Probability",
        "ML Risk Status",
    ]

    # Convert NaN / NumPy scalar values to JSON-safe Python values.
    records = table[display_columns].replace(
        {np.nan: None}
    ).to_dict(orient="records")

    return records


# =========================================================
# MAIN PAGE
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# DASHBOARD API
# =========================================================

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    global df

    if df.empty:
        return jsonify({
            "loaded": False,
            "message": "Upload your inventory CSV to get started.",
            "metrics": None,
            "status_counts": {},
            "trend": [],
            "inventory": [],
        })

    try:
        metrics = get_dashboard_metrics(df)
        table = get_inventory_table(df)

        # The donut chart displays ML prediction categories.
        status_counts = {
            "Low Risk": int(
                (df["Stockout Risk Prediction"] == 0).sum()
            ),
            "At Risk": int(
                (df["Stockout Risk Prediction"] == 1).sum()
            ),
        }

        trend = []

        trend_data = df[["Date", "Units Sold"]].copy()
        trend_data["Date"] = pd.to_datetime(
            trend_data["Date"], errors="coerce"
        )
        trend_data = trend_data.dropna(
            subset=["Date", "Units Sold"]
        )

        if not trend_data.empty:
            grouped = (
                trend_data.groupby("Date")["Units Sold"]
                .sum()
                .sort_index()
                .tail(30)
            )

            trend = [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "units": float(value),
                }
                for date, value in grouped.items()
            ]

        return jsonify({
            "loaded": True,
            "metrics": metrics,
            "status_counts": status_counts,
            "trend": trend,
            "inventory": table[:100],
        })

    except Exception as error:
        app.logger.exception("Dashboard error")
        return jsonify({
            "error": f"Unable to build dashboard: {str(error)}"
        }), 500


# =========================================================
# CSV UPLOAD API
# =========================================================

@app.route("/api/upload", methods=["POST"])
def upload():
    global df

    if "file" not in request.files:
        return jsonify({
            "error": "Please select a CSV file."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "error": "No file selected."
        }), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:
        uploaded_data = pd.read_csv(file)

        # Validate original columns and engineer missing features.
        prepared_data = prepare_data(uploaded_data)

        # Generate ML predictions using the saved model.
        analyzed_data = predict_stockout_risk(prepared_data)

        # Assign only after every stage succeeds.
        # A failed upload won't replace the previously loaded dataset.
        df = analyzed_data

        return jsonify({
            "success": True,
            "message": "Dataset analyzed successfully.",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        })

    except Exception as error:
        app.logger.exception("CSV upload or analysis failed")

        return jsonify({
            "error": str(error)
        }), 400


# =========================================================
# EXPORT ANALYZED DATASET
# =========================================================

@app.route("/api/export", methods=["GET"])
def export():
    global df

    if df.empty:
        return jsonify({
            "error": "Upload and analyze a dataset before exporting."
        }), 400

    try:
        report = add_operational_status(df)

        buffer = io.BytesIO()

        report.to_csv(
            buffer,
            index=False,
            encoding="utf-8-sig"
        )

        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name="stockwise_inventory_report.csv",
        )

    except Exception as error:
        app.logger.exception("Export failed")

        return jsonify({
            "error": f"Unable to export report: {str(error)}"
        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
