# ETA AI — Full Project Documentation

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Component Breakdown](#component-breakdown)
  - [1. Data Generation](#1-data-generation---generate_datapy)
  - [2. Model Training](#2-model-training---train_modelpy)
  - [3. API Service](#3-api-service---apppy)
- [How to Run](#how-to-run)
- [API Reference](#api-reference)
- [Concepts Explained](#concepts-explained)
- [What We Built & Achieved](#what-we-built--achieved)
- [Pros of This Architecture](#pros-of-this-architecture)
- [Future Improvements](#future-improvements)

---

## Overview

**ETA AI** is a machine learning microservice that predicts **Estimated Time of Arrival (ETA)**.

You provide:

- Distance (km)
- Speed (km/h)
- Hour of the day (0–23)
- Day of the week (0=Monday – 6=Sunday)

It returns an **ETA in hours**, factoring in traffic patterns like rush hours.

---

## Project Structure

```
eta_ai/
├── data/
│   ├── generate_data.py        # Synthetic training data generator
│   └── training_data.csv       # Generated training dataset (after running)
├── models/
│   └── eta_model.pkl           # Trained model artifact (after training)
├── service/
│   └── app.py                  # FastAPI prediction service
├── training/
│   └── train_model.py          # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Quick-start guide
└── DOCUMENTATION.md            # This file
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ETA AI PIPELINE                              │
│                                                                     │
│  ┌───────────────┐    ┌────────────────┐    ┌───────────────────┐  │
│  │  STEP 1       │    │  STEP 2        │    │  STEP 3           │  │
│  │  Data Layer   │───▶│  Training      │───▶│  Serving Layer    │  │
│  │               │    │  Layer          │    │                   │  │
│  │  generate_    │    │  train_model   │    │  FastAPI app      │  │
│  │  data.py      │    │  .py           │    │  (app.py)         │  │
│  │               │    │                │    │                   │  │
│  │  Output:      │    │  Output:       │    │  Endpoints:       │  │
│  │  CSV file     │    │  .pkl model    │    │  /health          │  │
│  │  (5000 rows)  │    │  (Random       │    │  /predict_eta     │  │
│  │               │    │   Forest)      │    │                   │  │
│  └───────────────┘    └────────────────┘    └───────────────────┘  │
│                                                                     │
│  Client (curl/browser/app)                                          │
│       │                                                             │
│       ▼                                                             │
│  POST /predict_eta                                                  │
│  { distance, speed, hour, weekday }                                 │
│       │                                                             │
│       ▼                                                             │
│  Response: { estimated_time_of_arrival: 0.30, unit: "hours" }      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Data Generation — `generate_data.py`

**Purpose:** Create synthetic but realistic travel data for training.

**Location:** `data/generate_data.py`

#### What It Does

Since we don't have real GPS or traffic data, we simulate realistic travel scenarios.
For each of the 5000 rows, the script:

1. Picks a **random distance** between 0.5 and 10 km
2. Picks a **random speed** between 10 and 40 km/h
3. Picks a **random hour** between 0 and 23
4. Picks a **random weekday** between 0 and 6
5. Calculates a **traffic factor**:
   - Rush hours (8–10 AM, 5–7 PM) → **1.5x** slower
   - Normal hours → **1.0x** (no change)
6. Computes ETA: `(distance / speed) × traffic_factor`

#### Code Walkthrough

```python
np.random.seed(42)  # Makes randomness reproducible — same data every run
```

```python
distance = np.random.uniform(0.5, 10)    # Float between 0.5–10
speed = np.random.uniform(10, 40)         # Float between 10–40
hour = np.random.randint(0, 24)           # Integer 0–23
weekday = np.random.randint(0, 7)         # Integer 0–6
```

```python
# The KEY pattern the model will learn:
traffic_factor = 1.5 if 8 <= hour <= 10 or 17 <= hour <= 19 else 1.0
eta = (distance / speed) * traffic_factor
```

#### Example Output

| distance | speed | hour | weekday | eta    |
| -------- | ----- | ---- | ------- | ------ |
| 5.00     | 25.0  | 9    | 1       | 0.3000 |
| 5.00     | 25.0  | 14   | 3       | 0.2000 |
| 2.50     | 20.0  | 18   | 5       | 0.1875 |
| 8.00     | 35.0  | 3    | 0       | 0.2286 |

> Notice: Same distance & speed, but **rush hour (9 AM)** produces a **50% higher ETA** than normal hours (2 PM). This is the pattern the model learns.

#### CLI Arguments

| Argument   | Default                  | Description          |
| ---------- | ------------------------ | -------------------- |
| `--rows`   | `5000`                   | Number of data rows  |
| `--output` | `data/training_data.csv` | Output CSV file path |

---

### 2. Model Training — `train_model.py`

**Purpose:** Read the CSV, train a Random Forest model, evaluate it, and save it to disk.

**Location:** `training/train_model.py`

#### What It Does

```
CSV File (5000 rows)
       │
       ▼
┌──────────────────────────┐
│ Extract Features & Target │
│   X = [distance, speed,   │
│        hour, weekday]      │
│   Y = [eta]                │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Train/Test Split          │
│   80% → training set      │
│   20% → testing set       │
│   random_state=42          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Train RandomForest        │
│   100 decision trees      │
│   Each tree learns        │
│   patterns independently  │
│   n_jobs=-1 (all cores)   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Evaluate on Test Data     │
│   MAE  = average error    │
│   R²   = accuracy score   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Save Model (.pkl file)    │
│   joblib.dump()           │
│   Binary serialization    │
└──────────────────────────┘
```

#### Code Walkthrough

```python
# Split data: 80% to learn from, 20% to test against
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
```

```python
# Create 100 decision trees that vote together
model = RandomForestRegressor(
    n_estimators=100,    # 100 trees
    random_state=42,     # Reproducible
    n_jobs=-1            # Use all CPU cores
)
model.fit(X_train, Y_train)  # Learn patterns
```

```python
# Test how good the model is
predictions = model.predict(X_test)
mae = mean_absolute_error(Y_test, predictions)  # Avg error in hours
r2 = r2_score(Y_test, predictions)              # 1.0 = perfect
```

```python
# Save the trained model to a binary file
joblib.dump(model, model_path)
```

#### Evaluation Metrics

| Metric  | What It Means                        | Good Value   | Expected |
| ------- | ------------------------------------ | ------------ | -------- |
| **MAE** | Average prediction error in hours    | Close to 0   | ~0.003   |
| **R²**  | How much variance the model explains | Close to 1.0 | ~0.99    |

#### CLI Arguments

| Argument       | Default                  | Description            |
| -------------- | ------------------------ | ---------------------- |
| `--data`       | `data/training_data.csv` | Path to training CSV   |
| `--model`      | `models/eta_model.pkl`   | Output model file path |
| `--estimators` | `100`                    | Number of trees        |

---

### 3. API Service — `app.py`

**Purpose:** Wrap the trained model in a REST API for real-time predictions.

**Location:** `service/app.py`

#### What It Does

```
Server starts (uvicorn)
       │
       ▼
┌──────────────────────┐
│ lifespan() - STARTUP │
│                      │
│ joblib.load()        │  Load .pkl model into memory
│ ml_model["eta"]      │  Store in global dict
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ SERVER RUNNING       │
│ Listening on :8000   │
│                      │
│ Endpoints:           │
│   GET  /health       │  → Health check
│   POST /predict_eta  │  → ETA prediction
│   GET  /docs         │  → Swagger UI (auto-generated)
└──────────┬───────────┘
           │  (on shutdown)
           ▼
┌──────────────────────┐
│ lifespan() - CLEANUP │
│ ml_model.clear()     │  Free memory
└──────────────────────┘
```

#### Request Flow for `/predict_eta`

```
Client sends POST request
       │
       ▼
┌──────────────────────┐
│ Pydantic Validation   │
│                       │
│ distance > 0?    ✓/✗  │
│ speed > 0?       ✓/✗  │
│ 0 ≤ hour ≤ 23?  ✓/✗  │
│ 0 ≤ weekday ≤ 6? ✓/✗ │
│                       │
│ If invalid → 422 error│
└──────────┬───────────┘
           │ (valid)
           ▼
┌──────────────────────┐
│ model.predict()       │
│                       │
│ Input:  [[5, 30, 9, 2]]│
│ Output: [0.2987...]   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Return JSON Response  │
│ {                     │
│   "estimated_time_    │
│    of_arrival": 0.30, │
│   "unit": "hours"     │
│ }                     │
└──────────────────────┘
```

#### Code Walkthrough

**Model Loading (runs once at startup):**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["eta"] = joblib.load(MODEL_PATH)   # Load model
    yield                                         # Server runs
    ml_model.clear()                              # Cleanup on shutdown
```

**Input Validation (automatic via Pydantic):**

```python
class ETARequest(BaseModel):
    distance: float = Field(..., gt=0)           # Must be > 0
    speed:    float = Field(..., gt=0)           # Must be > 0
    hour:     int   = Field(..., ge=0, le=23)    # 0 to 23
    weekday:  int   = Field(..., ge=0, le=6)     # 0 to 6
```

> If someone sends `{"distance": -5}`, FastAPI **automatically rejects** it with a `422 Unprocessable Entity` error. You don't write a single `if` statement.

**Prediction Endpoint:**

```python
@app.post("/predict_eta", response_model=ETAResponse)
def predict_eta(data: ETARequest):
    features = [[data.distance, data.speed, data.hour, data.weekday]]
    eta = ml_model["eta"].predict(features)[0]
    return ETAResponse(estimated_time_of_arrival=round(float(eta), 2))
```

---

## How to Run

### Prerequisites

- Python 3.9+

### Step-by-Step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data
python data/generate_data.py --rows 5000 --output data/training_data.csv

# 3. Train the model
python training/train_model.py --data data/training_data.csv --model models/eta_model.pkl --estimators 100

# 4. Start the API server
uvicorn service.app:app --reload --host 0.0.0.0 --port 8000
```

### Test the API

**Health Check:**

```bash
curl http://localhost:8000/health
```

```json
{ "status": "healthy", "model_loaded": true }
```

**Predict ETA:**

```bash
curl -X POST http://localhost:8000/predict_eta ^
  -H "Content-Type: application/json" ^
  -d "{\"distance\": 5.0, \"speed\": 30.0, \"hour\": 9, \"weekday\": 2}"
```

```json
{ "estimated_time_of_arrival": 0.25, "unit": "hours" }
```

**Swagger UI (Interactive Docs):**
Open `http://localhost:8000/docs` in your browser.

---

## API Reference

### `GET /health`

Returns the health status of the service.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict_eta`

Returns an ETA prediction based on input features.

**Request Body:**

| Field      | Type    | Constraints | Description                      |
| ---------- | ------- | ----------- | -------------------------------- |
| `distance` | `float` | `> 0`       | Distance in kilometers           |
| `speed`    | `float` | `> 0`       | Speed in km/h                    |
| `hour`     | `int`   | `0–23`      | Hour of the day                  |
| `weekday`  | `int`   | `0–6`       | Day of week (0=Monday, 6=Sunday) |

**Response:**

| Field                       | Type    | Description      |
| --------------------------- | ------- | ---------------- |
| `estimated_time_of_arrival` | `float` | Predicted ETA    |
| `unit`                      | `str`   | Always `"hours"` |

**Error Responses:**

| Code | When                             |
| ---- | -------------------------------- |
| 422  | Invalid input (Pydantic rejects) |
| 500  | Model prediction fails           |

---

## Concepts Explained

### What Is Random Forest?

Imagine asking **100 experts** (decision trees) to predict your travel time. Each expert looks at the data slightly differently. Then you **average all 100 answers**:

```
Tree 1:   "ETA = 0.28 hrs"
Tree 2:   "ETA = 0.31 hrs"
Tree 3:   "ETA = 0.29 hrs"
  ...
Tree 100: "ETA = 0.30 hrs"
──────────────────────────
Average →  ETA = 0.30 hrs  ✅
```

This ensemble approach is **more accurate and stable** than any single decision tree.

### What Is Pydantic?

A data validation library. Instead of writing:

```python
# Without Pydantic (tedious and error-prone)
if distance <= 0:
    raise ValueError("distance must be positive")
if hour < 0 or hour > 23:
    raise ValueError("hour must be 0-23")
```

You write:

```python
# With Pydantic (clean and automatic)
class ETARequest(BaseModel):
    distance: float = Field(..., gt=0)
    hour: int = Field(..., ge=0, le=23)
```

### What Is FastAPI Lifespan?

A way to run code **once at startup** and **once at shutdown**:

```python
@asynccontextmanager
async def lifespan(app):
    # STARTUP — runs once when server starts
    load_model()
    yield
    # SHUTDOWN — runs once when server stops
    cleanup()
```

This is better than loading the model on every request (slow) or at import time (uncontrollable).

### What Is joblib?

A library that **serializes** (saves) Python objects to binary files:

```python
joblib.dump(model, "model.pkl")   # Save trained model to disk
model = joblib.load("model.pkl")  # Load it back into memory
```

The `.pkl` file contains the entire trained Random Forest — all 100 trees, all learned thresholds — in a compact binary format.

---

## What We Built & Achieved

| Goal                    | Status | How                                             |
| ----------------------- | ------ | ----------------------------------------------- |
| End-to-end ML pipeline  | ✅     | Data → Training → Serving, all scripted         |
| Real-time predictions   | ✅     | FastAPI responds in milliseconds                |
| Input validation        | ✅     | Pydantic rejects bad data with 422 errors       |
| Error handling          | ✅     | try/except + HTTP 500 + structured logging      |
| Reproducibility         | ✅     | `random_state=42`, `np.random.seed(42)`         |
| Clean startup/shutdown  | ✅     | `lifespan` context manager                      |
| Configurable            | ✅     | CLI args (`argparse`) + env vars (`MODEL_PATH`) |
| Auto-generated API docs | ✅     | Swagger UI at `/docs` for free                  |
| Logging                 | ✅     | Python `logging` module on every prediction     |
| Multi-core training     | ✅     | `n_jobs=-1` uses all CPU cores                  |

---

## Pros of This Architecture

| Pro                        | Explanation                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| **Separation of Concerns** | Data, training, and serving are independent scripts — change one without breaking others |
| **Stateless API**          | Model loaded once at startup; each request is independent — easy to horizontally scale   |
| **FastAPI Performance**    | One of the fastest Python frameworks (async, Starlette + Uvicorn)                        |
| **Auto Documentation**     | Visit `/docs` → interactive Swagger UI generated from your code                          |
| **Validation for Free**    | Pydantic catches bad input before it touches your model                                  |
| **Portable Model**         | `.pkl` file can be deployed anywhere — Docker, cloud, edge device                        |
| **Structured Logging**     | Every prediction and error is logged for production debugging                            |
| **Reproducible Results**   | Fixed seeds mean the same data and model every single time                               |
| **Multi-Core Training**    | `n_jobs=-1` parallelizes across all CPU cores                                            |
| **Clean Lifecycle**        | `lifespan` ensures proper startup loading and shutdown cleanup                           |

---

## Future Improvements

| Current State       | Next Step                                 |
| ------------------- | ----------------------------------------- |
| Single process      | Docker + Kubernetes (multiple replicas)   |
| In-memory model     | Model registry (MLflow, W&B)              |
| No monitoring       | Prometheus + Grafana metrics              |
| Synthetic data      | Real traffic data (Google Maps API, Uber) |
| Single model        | A/B testing between model versions        |
| No CI/CD            | GitHub Actions pipeline                   |
| No caching          | Redis cache for repeated queries          |
| No authentication   | API key / OAuth2 authentication           |
| No rate limiting    | Rate limiting middleware                  |
| No containerization | Dockerfile + docker-compose               |

---

## Tech Stack

| Technology       | Role                             |
| ---------------- | -------------------------------- |
| **Python 3.9+**  | Programming language             |
| **FastAPI**      | REST API framework               |
| **scikit-learn** | Machine learning (Random Forest) |
| **pandas**       | Data manipulation                |
| **numpy**        | Numerical operations             |
| **joblib**       | Model serialization              |
| **uvicorn**      | ASGI server                      |
| **Pydantic**     | Data validation                  |

---

_Built as a learning project demonstrating the full ML pipeline — from data generation to model serving. The same architecture pattern is used by companies like Uber, Lyft, and DoorDash for their ETA services, at a larger scale._
