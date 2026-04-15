# Heart Disease Prediction - ML Course Project

A complete machine learning pipeline for predicting heart disease risk using clinical features. Includes interactive web app, model dashboard, and containerized deployment.

## Project Overview

**Dataset:** Cleveland Heart Disease Database (303 unique patients)  
**Target:** Binary classification (No Disease: 0, Disease: 1)  
**Features:** 13 clinical measurements (age, sex, chest pain type, blood pressure, etc.)  
**Best Model:** Tuned SVM with 80.3% test accuracy and 88% recall for disease detection

### Key Finding: Data Quality Issue

The original replicated dataset contained **70.5% duplicate rows**, which inflated Random Forest accuracy to 99.7% via data leakage. The analysis uncovered this and evaluated models on both the full and deduplicated (302 unique) datasets.

## Files Structure

```
archive/
├── app.py                          # Streamlit prediction web app
├── dashboard.py                    # Dash interactive model dashboard
├── Dockerfile                      # Docker containerization
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Python dependencies
├── Heart_disease_cleveland_new.csv # Original dataset (303 rows)
└── README.md                       # This file
```

## Installation & Usage

### Option 1: Local Development

#### Prerequisites
- Python 3.8+
- pip

#### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Run Streamlit App (Prediction)
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

**Features:**
- Interactive patient data input (13 clinical features)
- Real-time risk prediction
- Visual risk gauge and probability charts
- Model information and interpretation guide
- Patient summary table

#### Run Dash Dashboard (Model Analysis)
```bash
python dashboard.py
```
Access at: `http://localhost:8050`

**Features:**
- Model comparison (5-fold cross-validation)
- ROC curve with AUC score
- Confusion matrix heatmap
- Classification metrics (accuracy, precision, recall, F1)
- Feature importance rankings
- Class distribution visualization

### Option 2: Docker Deployment

#### Prerequisites
- Docker
- Docker Compose

#### Build & Run
```bash
# Build images
docker-compose build

# Start all services
docker-compose up
```

**Access:**
- Streamlit App: `http://localhost:8501`
- Dash Dashboard: `http://localhost:8050`

#### Stop Services
```bash
docker-compose down
```

#### Run Individual Services
```bash
# Just the Streamlit app
docker run -p 8501:8501 heart-disease-predictor streamlit run app.py

# Just the Dash dashboard
docker run -p 8050:8050 heart-disease-dashboard python dashboard.py
```

## Model Details

### Algorithm: Support Vector Machine (SVM)
- **Kernel:** Linear
- **Regularization (C):** 0.1
- **Scaling:** StandardScaler
- **Hyperparameter Tuning:** GridSearchCV (5-fold CV)

### Performance Metrics (Test Set, 80/20 Split)
| Metric | Score |
|--------|-------|
| Accuracy | 80.3% |
| Precision (Disease) | 78% |
| Recall (Disease) | 88% |
| F1-Score | 83% |
| ROC-AUC | 0.947 |

**Why high recall matters:** In medical screening, missing a disease case (false negative) is more costly than a false alarm. 88% recall means we catch 88% of disease cases.

### Top 5 Predictive Features
1. **Chest Pain Type (cp)** — 18.2% importance
2. **Max Heart Rate (thalach)** — 14.5%
3. **Major Vessels (ca)** — 11.3%
4. **ST Depression (oldpeak)** — 9.8%
5. **Thalassemia (thal)** — 9.0%

## Feature Descriptions

| Feature | Range | Description |
|---------|-------|-------------|
| **age** | 29-77 years | Patient age |
| **sex** | 0/1 | 0=Female, 1=Male |
| **cp** | 0-3 | Chest pain type (0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic) |
| **trestbps** | 94-192 mmHg | Resting blood pressure |
| **chol** | 126-353 mg/dl | Serum cholesterol |
| **fbs** | 0/1 | Fasting blood sugar > 120 mg/dl (0=No, 1=Yes) |
| **restecg** | 0-2 | Resting ECG results |
| **thalach** | 71-202 bpm | Maximum heart rate achieved |
| **exang** | 0/1 | Exercise-induced angina (0=No, 1=Yes) |
| **oldpeak** | 0-5.6 | ST depression (exercise vs rest) |
| **slope** | 1-3 | Slope of peak exercise ST segment (1=Upsloping, 2=Flat, 3=Downsloping) |
| **ca** | 0-4 | Number of major vessels (0-4) colored by fluoroscopy |
| **thal** | 0-3 | Thalassemia type (0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Normal variant) |

## Important Disclaimers

⚠️ **This model is for educational and demonstration purposes ONLY.**

- **Not for clinical use.** Do not rely on this model for medical diagnosis or treatment decisions.
- **Always consult healthcare professionals** for medical advice.
- **Model trained on limited data** (303 patients) from the 1980s Cleveland database.
- **Predictions are estimates** based on historical patterns; individual cases may vary significantly.

## Lessons Learned

### Data Quality is Critical
- **70.5% duplicates** inflated model accuracy from ~82% to 99.7%
- Always check for duplicates before cross-validation
- Data leakage is subtle but devastating

### Tree-Based vs Parametric Models
- **Random Forest:** Memorizes training data → highly vulnerable to leakage (+17.9% inflation)
- **SVM/Logistic Regression:** Learn global boundaries → less inflated (+8.6% inflation)

### Medical Classification is Different
- **Accuracy alone is misleading** when class imbalance exists
- **Recall matters more** — missing disease cases is costly
- **AUC-ROC is more robust** across threshold variations

## Technologies

- **Backend:** Python 3.9, scikit-learn, pandas
- **Frontend:** Streamlit (app.py), Dash/Plotly (dashboard.py)
- **Deployment:** Docker & Docker Compose
- **ML Techniques:** Cross-validation, hyperparameter tuning, ROC analysis, feature importance

## Future Enhancements

1. **Patient History Tracking** — Store predictions and track patient trends
2. **REST API** — Expose model via FastAPI for integration with EHR systems
3. **Model Retraining Pipeline** — Automated retraining on new data
4. **Confidence Intervals** — Quantify prediction uncertainty
5. **Explainability** — SHAP values for per-prediction feature attribution
6. **Cloud Deployment** — AWS/GCP/Azure integration for production use

## Author

Created for an ML course project analyzing heart disease classification.

## License

Educational use only.
