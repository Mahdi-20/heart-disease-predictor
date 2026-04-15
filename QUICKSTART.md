# Quick Start Guide

## 🚀 Get Running in 2 Minutes

### For Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit prediction app
streamlit run app.py

# 3. In another terminal, run the Dash dashboard
python dashboard.py
```

**Then open in browser:**
- 🔮 Prediction App: `http://localhost:8501`
- 📊 Model Dashboard: `http://localhost:8050`

---

### For Docker (All-in-One)

```bash
# 1. Start both services
docker-compose up

# 2. Open in browser:
#    - http://localhost:8501 (Streamlit)
#    - http://localhost:8050 (Dashboard)

# 3. Stop with Ctrl+C or:
docker-compose down
```

---

## 📋 What Each Tool Does

### **Streamlit App** (`app.py`)
- Enter patient clinical data (age, blood pressure, cholesterol, etc.)
- Get instant heart disease risk prediction (0-100%)
- See confidence score and visual risk gauge
- Read model information and interpretation guide

### **Dash Dashboard** (`dashboard.py`)
- Compare 4 different ML models
- View ROC curves and AUC scores
- Analyze feature importance
- See confusion matrices and classification metrics
- Understand model performance in detail

---

## ✅ Verification

**Streamlit working?** You should see:
- Header: "❤️ Heart Disease Risk Predictor"
- Sidebar with patient data inputs
- Three metric boxes showing Risk Score, Prediction, Confidence
- Probability bar chart and risk gauge
- Patient summary table

**Dashboard working?** You should see:
- 4 metric cards (Accuracy, ROC-AUC, Recall, Precision)
- 6 visualizations (CV comparison, ROC curve, confusion matrix, feature importance, etc.)
- All charts are interactive (hover for details, zoom, pan)

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| Port 8501 already in use | Change in app.py or: `streamlit run app.py --server.port 8502` |
| Port 8050 already in use | Change port in dashboard.py: `app.run_server(port=8051)` |
| CSV file not found | Ensure `Heart_disease_cleveland_new.csv` is in same directory as scripts |
| Docker image won't build | Run `docker-compose build --no-cache` |

---

## 📊 Expected Results

**Test Accuracy:** ~80% (varies by random seed)  
**ROC-AUC:** ~0.95 (excellent discrimination)  
**Recall for Disease:** ~88% (catches most disease cases)  
**Top Feature:** Chest pain type (18% importance)

---

## 🎯 Next Steps

1. **Try the app:** Enter patient data and explore predictions
2. **Study the dashboard:** Understand model strengths/weaknesses
3. **Review the data:** Check feature ranges in README.md
4. **Customize:** Modify sliders, colors, or models
5. **Deploy:** Use docker-compose for production-like environment

---

## 📚 Full Documentation

See **README.md** for:
- Detailed feature descriptions
- Model architecture and hyperparameters
- Important medical disclaimers
- Technology stack
- Future enhancement ideas
