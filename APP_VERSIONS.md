# Two App Versions: Choose Your Preference

Your project now includes **two separate Streamlit apps** with different capabilities. You can run either one depending on your needs.

---

## 📊 Quick Comparison

| Feature | `app.py` | `app_with_retraining.py` |
|---------|----------|--------------------------|
| **Core Prediction** | ✅ | ✅ |
| **Patient History** | ✅ | ✅ |
| **Risk Trends** | ✅ | ✅ |
| **Patient ID Support** | ✅ | ✅ |
| **SHAP Explainability** | ✅ | ✅ |
| **Model Retraining** | ❌ | ✅ |
| **Continuous Learning** | ❌ | ✅ |
| **Data File Size** | Smaller | Larger (13 features) |
| **Complexity** | Simple | Advanced |

---

## 🎯 app.py (Standard Version)

**Use this if:** You want a clean, simple app focused on predictions without model retraining.

### Features:
- Make heart disease predictions with clinical features
- Save predictions to patient history
- View risk trends for individual patients
- Compare multiple predictions for the same patient
- SHAP feature importance analysis
- Patient ID to distinguish same-name patients

### Patient History Structure:
Stores only essential data:
```json
{
  "timestamp": "2026-04-10T08:30:00",
  "patient_id": "P001",
  "patient_name": "John Smith",
  "prediction": 0,
  "risk_percentage": 23.5,
  "confidence": 0.78,
  "age": 45,
  "max_heart_rate": 185,
  "cholesterol": 180
}
```

### Files Used:
- `patient_history.json` - Stores prediction history
- `Heart_disease_cleveland_new.csv` - Original training data (303 patients)

### Run Command:
```bash
streamlit run app.py
```

---

## 🚀 app_with_retraining.py (Advanced Version)

**Use this if:** You want to continuously improve your model by retraining it with new patient data.

### Additional Features:
- **Model Retraining** - Click "🔄 Retrain Model with New Data" button
- **Continuous Learning** - Combine original data + your patient data
- **Cross-Validation Scores** - See CV accuracy after retraining
- **Session-based Model Switching** - Uses retrained model during current session
- **Model Status Indicator** - Know which model you're using

### Patient History Structure:
Stores **all 13 clinical features** for retraining:
```json
{
  "timestamp": "2026-04-10T08:30:00",
  "patient_id": "P001",
  "patient_name": "John Smith",
  "age": 45.0,
  "sex": 1.0,
  "cp": 0.0,
  "trestbps": 130.0,
  "chol": 180.0,
  "fbs": 0.0,
  "restecg": 0.0,
  "thalach": 185.0,
  "exang": 0.0,
  "oldpeak": 1.0,
  "slope": 1.0,
  "ca": 0.0,
  "thal": 0.0,
  "prediction": 0,
  "risk_percentage": 23.5,
  "confidence": 0.78,
  "target": 0
}
```

### Files Used:
- `patient_history_retraining.json` - Stores all features for retraining
- `Heart_disease_cleveland_new.csv` - Original training data (303 patients)

### Run Command:
```bash
streamlit run app_with_retraining.py
```

### How to Use Retraining:
1. Save 5+ patient predictions using the "💾 Save to History" button
2. Scroll down to **"🔧 Model Management & Retraining"** section
3. Click **"🔄 Retrain Model with New Data"**
4. See the CV accuracy and confirmation
5. All predictions now use the updated model

---

## 🔄 Comparison of Retraining Workflow

### Without Retraining (app.py):
```
Day 1: Save predictions → View in history → Done
Day 2: Save more predictions → View trends → Done
(Model never changes - always uses original 303 patients)
```

### With Retraining (app_with_retraining.py):
```
Day 1: Save 5 predictions
Day 2: Click "Retrain Model"
       → Model retrains on: 303 original + 5 new = 308 total
       → CV Accuracy: e.g., 84% (original) → 85% (retrained)
       → All predictions now use improved model
Day 3: Save 10 more predictions
Day 4: Click "Retrain Model" again
       → Model retrains on: 303 original + 15 new = 318 total
       → Continuous improvement!
```

---

## 💾 File Storage

### Standard Version (app.py):
```
D:\ML course\Our_project\Heart\archive\
├── app.py
├── patient_history.json                 (3 key fields)
└── Heart_disease_cleveland_new.csv      (original data)
```

### Advanced Version (app_with_retraining.py):
```
D:\ML course\Our_project\Heart\archive\
├── app_with_retraining.py
├── patient_history_retraining.json      (13 features)
└── Heart_disease_cleveland_new.csv      (original data)
```

**Important:** Each app uses its own history file, so they don't interfere with each other.

---

## 🎓 For Your Course Project

### Recommended Approach:

**Presentation to Professor:**
- Use `app.py` for **clean, professional presentations**
- Shows: predictions, patient history, trends, SHAP analysis
- Focus on current state predictions without data complications

**Exploring Advanced Concepts:**
- Use `app_with_retraining.py` to **demonstrate model improvement**
- Show how models evolve with more data
- Discuss continuous learning in machine learning
- Use with **clean, realistic data only**

---

## ⚠️ Important Notes

### About Retraining Data:
- ✅ **Good data**: Accurate patient measurements and confirmed diagnoses
- ❌ **Bad data**: Incorrect values, duplicates, or unconfirmed predictions
- If your history has bad data, stick with `app.py` (no retraining)
- If your history has good data, use `app_with_retraining.py` to improve the model

### Session-Only Models:
- Retrained models are stored in browser memory only
- They reset when you refresh the page
- This is intentional to prevent accidental permanent changes
- If you want to save models permanently, you'd need disk persistence (future enhancement)

### Minimum Requirements for Retraining:
- At least 5 patient records must be saved
- Each record must have all 13 features
- Both conditions are checked before allowing retraining

---

## 🚀 Quick Start

**Want just predictions with history?**
```bash
streamlit run app.py
```

**Want to improve the model with new data?**
```bash
streamlit run app_with_retraining.py
```

Both apps have:
- ✅ Two-tab interface (Prediction + History)
- ✅ Patient ID field
- ✅ Risk trends analysis
- ✅ SHAP explainability
- ✅ Professional styling

---

## 📝 Summary

| Need | Use |
|------|-----|
| Clean presentation | `app.py` |
| Demonstrate model improvement | `app_with_retraining.py` |
| Teaching predictions | `app.py` |
| Research continuous learning | `app_with_retraining.py` |
| Production-ready | Neither (educational only) |

Both are valid for your course project. Choose based on your presentation goals! 🎯
