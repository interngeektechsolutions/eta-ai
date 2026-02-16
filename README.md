# ETA AI - Estimated Time of Arrival Predictor

An AI-powered microservice that predicts Estimated Time of Arrival (ETA) using a Random Forest Regression model. The service takes into account distance, speed, time of day, and day of week — including traffic patterns during rush hours.

## Project Structure

```
eta_ai/
├── data/
│   ├── generate_data.py        # Synthetic training data generator
│   └── training_data.csv       # Generated training dataset
├── models/
│   └── eta_model.pkl           # Trained model artifact
├── service/
│   └── app.py                  # FastAPI prediction service
├── training/
│   └── train_model.py          # Model training script
├── requirements.txt
└── README.md
```

## Features

- **Synthetic Data Generation**: Generates 5,000 rows of realistic ETA data with traffic factor simulation (1.5x during rush hours: 8–10 AM & 5–7 PM).
- **Random Forest Model**: Trained with 100 estimators on distance, speed, hour, and weekday features.
- **FastAPI Service**: Lightweight REST API for real-time ETA predictions.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python data/generate_data.py
```

### 3. Train the Model

```bash
python training/train_model.py
```

### 4. Run the API Server

```bash
uvicorn service.app:app --reload
```

### 5. Make a Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict_eta \
  -H "Content-Type: application/json" \
  -d '{"distance": 5.0, "speed": 25.0, "hour": 9, "weekday": 1}'
```

**Response:**

```json
{
  "estimated_time_of_arrival": 0.3
}
```

## API Reference

### `POST /predict_eta`

| Parameter  | Type    | Description                      |
| ---------- | ------- | -------------------------------- |
| `distance` | `float` | Distance in kilometers           |
| `speed`    | `float` | Speed in km/h                    |
| `hour`     | `int`   | Hour of day (0–23)               |
| `weekday`  | `int`   | Day of week (0=Monday, 6=Sunday) |

## Tech Stack

- **Python**
- **FastAPI** — REST API framework
- **scikit-learn** — Machine learning (Random Forest)
- **pandas / numpy** — Data processing
- **joblib** — Model serialization
- **uvicorn** — ASGI server# ETA AI - Estimated Time of Arrival Predictor

An AI-powered microservice that predicts Estimated Time of Arrival (ETA) using a Random Forest Regression model. The service takes into account distance, speed, time of day, and day of week — including traffic patterns during rush hours.

## Project Structure

```
eta_ai/
├── data/
│   ├── generate_data.py        # Synthetic training data generator
│   └── training_data.csv       # Generated training dataset
├── models/
│   └── eta_model.pkl           # Trained model artifact
├── service/
│   └── app.py                  # FastAPI prediction service
├── training/
│   └── train_model.py          # Model training script
├── requirements.txt
└── README.md
```

## Features

- **Synthetic Data Generation**: Generates 5,000 rows of realistic ETA data with traffic factor simulation (1.5x during rush hours: 8–10 AM & 5–7 PM).
- **Random Forest Model**: Trained with 100 estimators on distance, speed, hour, and weekday features.
- **FastAPI Service**: Lightweight REST API for real-time ETA predictions.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python data/generate_data.py
```

### 3. Train the Model

```bash
python training/train_model.py
```

### 4. Run the API Server

```bash
uvicorn service.app:app --reload
```

### 5. Make a Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict_eta \
  -H "Content-Type: application/json" \
  -d '{"distance": 5.0, "speed": 25.0, "hour": 9, "weekday": 1}'
```

**Response:**

```json
{
  "estimated_time_of_arrival": 0.3
}
```

## API Reference

### `POST /predict_eta`

| Parameter  | Type    | Description                      |
| ---------- | ------- | -------------------------------- |
| `distance` | `float` | Distance in kilometers           |
| `speed`    | `float` | Speed in km/h                    |
| `hour`     | `int`   | Hour of day (0–23)               |
| `weekday`  | `int`   | Day of week (0=Monday, 6=Sunday) |

## Tech Stack

- **Python**
- **FastAPI** — REST API framework
- **scikit-learn** — Machine learning (Random Forest)
- **pandas / numpy** — Data processing
- **joblib** — Model serialization
- **uvicorn** — ASGI server
