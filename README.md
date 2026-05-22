# Adult Income Prediction

This repository contains a complete deployment-ready project for predicting whether an adult earns `>50K` using the UCI Adult dataset.

## What is included

- `app.py` - FastAPI service for dataset metadata, statistics, supported categorical values, and income prediction.
- `streamlit_app.py` - Streamlit UI to send prediction requests to the FastAPI backend.
- `Dockerfile` - Docker configuration to run the FastAPI application.
- `requirements.txt` - Python dependencies required for API, UI, and model loading.
- `doctor.py` - Health-check script to validate dataset, model, and prediction pipeline.
- `adult.csv` - Dataset used for training and API category mappings.
- `model.pkl` - Saved SVC model pipeline trained on cleaned Adult dataset.

## Model performance

The following accuracies were measured using the same preprocessing flow found in the notebook and dataset:

- `SVC` (saved model): **0.8526**
- `Logistic Regression`: **0.8302**
- `Decision Tree`: **0.8152**
- `KNN`: **0.8328**

> The final deployed model is the saved `SVC` pipeline, and the API includes the same label-encoding mapping that was used during training.

## API endpoints

- `GET /` - Service overview
- `GET /health` - Health check and model status
- `GET /metadata` - Dataset shape and income labels
- `GET /supported-values` - All supported categorical values for prediction
- `GET /sample?limit=10` - Sample dataset rows
- `GET /statistics` - Dataset counts and averages
- `POST /predict` - Make a prediction

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "Private",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "educational-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "gender": "Female",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }'
```

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run FastAPI:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3. Run Streamlit UI:

```bash
streamlit run streamlit_app.py
```

4. Run diagnostic checks:

```bash
python doctor.py
```

## Docker usage

Build the image:

```bash
docker build -t adult-income-api .
```

Run the container:

```bash
docker run -p 8000:8000 adult-income-api
```

Then visit `http://localhost:8000/docs` for the FastAPI interactive docs.

## Hugging Face deployment

Use the `Dockerfile`, `app.py`, and `requirements.txt` to deploy on Hugging Face Spaces or any container-based platform.

- Example HF Space URL placeholder: `https://huggingface.co/spaces/<your-username>/<your-space>`
- Example GitHub placeholder: `https://github.com/<your-username>/<your-repo>`

## Notes

- The API now accepts categorical values directly and applies the same label-encoding mapping as training.
- `streamlit_app.py` provides a clean form with each supported option.
- The FastAPI app exposes `/supported-values` so the UI or external integrations can see valid category values.
