# Model Retraining Guide

## How to Retrain Your Model with New Patient Data

### Overview
Your Streamlit app now supports **continuous learning**. As new patient data is collected and saved to history, you can retrain the model to improve predictions.

---

## Workflow

### Step 1: Collect Patient Data
1. Use the **"🔮 Make Prediction"** tab to make predictions for new patients
2. Enter all required patient information in the sidebar
3. Enter a **Patient ID** (required, e.g., P001, P002)
4. Click **"💾 Save to History"** button
5. The complete patient data (all 13 features) is stored in `patient_history.json`

### Step 2: Retrain the Model
1. Go to the **"🔮 Make Prediction"** tab
2. Scroll down to find **"🔧 Model Management & Retraining"** section
3. Click the **blue button "🔄 Retrain Model with New Data"**
4. Wait for the model to retrain (combines original 303 patients + your new data)
5. You'll see:
   - ✅ Success message with new accuracy metrics
   - 🎉 Balloons animation
   - Patient record count

### Step 3: Use the Retrained Model
- After retraining, a green info box appears: **"✅ USING RETRAINED MODEL"**
- All predictions now use the updated model
- This persists during your current session
- If you refresh the browser, it reverts to the original model (retrained models are session-only)

---

## Data Requirements

### Minimum Data for Retraining
- **At least 5 patient records** must be saved before retraining
- Each record must include all **13 features**:
  1. Age
  2. Sex (0=Female, 1=Male)
  3. Chest Pain Type (0-3)
  4. Resting Blood Pressure
  5. Serum Cholesterol
  6. Fasting Blood Sugar (0/1)
  7. Resting ECG Results (0-2)
  8. Max Heart Rate Achieved
  9. Exercise-Induced Angina (0/1)
  10. ST Depression
  11. Slope of ST Segment (1-3)
  12. Major Vessels Colored (0-4)
  13. Thalassemia Type (0-3)

### What Gets Retrained
- **Original dataset**: 303 patients from Cleveland Heart Disease dataset
- **Your new data**: All records saved in `patient_history.json`
- **Combined model**: Trained on original + new data using 5-fold cross-validation

---

## Features of Retraining

### ✅ Automatic Feature Saving
- When you save a prediction, **all 13 input features** are automatically stored
- No manual data entry needed for retraining

### ✅ Cross-Validation
- Uses 5-fold cross-validation to evaluate the retrained model
- Shows CV accuracy (mean ± std deviation)

### ✅ Data Persistence
- Patient history saved in `patient_history.json`
- Each patient record includes timestamp, ID, name, and all features

### ✅ Model Status Indicator
- Green info box shows if retrained model is active
- Blue box shows if original model is being used

### ⚠️ Session-Based Retraining
- Retrained models only persist during current session
- Browser refresh reverts to original model
- To permanently save a model, save the trained model file to disk (future enhancement)

---

## Example Scenario

**Day 1**: Save 5 patient predictions → Not enough to retrain yet

**Day 2**: Save 3 more predictions (8 total) → Now you can retrain!
- Click **"🔄 Retrain Model with New Data"**
- Model retrains on: 303 (original) + 8 (new) = 311 total patients
- CV Accuracy might improve (e.g., 83% → 85%)

**Day 3**: Save more predictions → Click retrain again
- Model retrains on: 303 (original) + 15 (new) = 318 total patients
- Performance continues to improve with more real data

---

## Technical Details

### Model Architecture
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Linear
- **Regularization**: C=0.1
- **Probability**: Enabled for confidence scores

### Data Handling
- Features are **StandardScaler normalized** separately for retraining
- Original dataset scaler is NOT used (new scaler fitted on combined data)
- Missing features are automatically skipped (robust error handling)

### Performance Metrics
- **Accuracy**: Overall correctness
- **CV Scores**: Show model stability across folds
- **Confidence**: Probability of prediction for each case

---

## Troubleshooting

### "Not enough patient data to retrain (need at least 5 records)"
- Save at least 5 patient predictions before attempting to retrain
- Check `patient_history.json` to verify records are saved

### Retrained model reverted after page refresh
- This is expected! Retrained models are session-only
- To keep improvements, implement model persistence (save to disk)

### Predictions haven't changed after retraining
- Some new data might confirm existing patterns
- More diverse data improves model generalization

---

## Future Enhancements

1. **Persistent Model Saving**: Save retrained models to disk with versioning
2. **A/B Testing**: Compare original vs retrained model accuracy on new data
3. **Scheduled Retraining**: Automatic retraining on a set schedule
4. **Feature Importance**: Show which new patients most impact predictions
5. **Model Versioning**: Keep track of multiple model versions

---

## File Structure

```
D:\ML course\Our_project\Heart\archive\
├── app.py                      # Main Streamlit app (with retraining)
├── patient_history.json        # Stores all patient predictions
├── Heart_disease_cleveland_new.csv  # Original training data
├── RETRAINING_GUIDE.md         # This file
└── [other files]
```

---

## Questions?

Check the **"ℹ️ About This Model"** section in the app for model details!
