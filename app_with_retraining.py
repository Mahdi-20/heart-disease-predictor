"""
Heart Disease Prediction Web App - WITH RETRAINING
Interactive Streamlit application with patient history tracking, model retraining, and SHAP explainability
Includes advanced features for continuous learning with new patient data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import shap
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Heart Disease Predictor - Advanced",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #065A82;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .metric-box {
        background-color: #EBF4FA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #028090;
        text-align: center;
    }
    .risk-high {
        color: #C0392B;
        font-weight: bold;
    }
    .risk-low {
        color: #1E7E34;
        font-weight: bold;
    }
    /* Tab styling */
    [role="tablist"] {
        background: linear-gradient(135deg, #028090 0%, #00A896 100%);
        padding: 10px;
        border-radius: 10px;
        gap: 10px;
    }
    [role="tab"] {
        font-size: 1000px !important;
        font-weight: 1000 !important;
        padding: 12px 24px !important;
        background-color: rgba(255, 255, 0, 0.2) !important;
        color: white !important;
        border-radius: 8px !important;
        border: 2px solid transparent !important;
        transition: all 0.3s ease;
    }
    [role="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px);
    }
    [role="tab"][aria-selected="true"] {
        background-color: white !important;
        color: #028090 !important;
        border: 2px solid #028090 !important;
        font-weight: 800 !important;
    }
    /* Save to History button styling */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border: 2px solid #27AE60 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #1E8449 0%, #27AE60 100%) !important;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4) !important;
    }
    button[kind="secondary"]:active {
        transform: scale(0.98);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>❤️ Heart Disease Risk Predictor (Advanced)</h1>", unsafe_allow_html=True)
st.markdown("**AI-powered prediction with continuous learning** | Original Cleveland Dataset (303 patients) + Your Data")

# Initialize session state for model management
if 'use_retrained' not in st.session_state:
    st.session_state.use_retrained = False
if 'retrained_model' not in st.session_state:
    st.session_state.retrained_model = None
if 'retrained_scaler' not in st.session_state:
    st.session_state.retrained_scaler = None

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Patient History & Trends", "ℹ️ About Project"])

# ============================================================================
# PATIENT HISTORY MANAGEMENT
# ============================================================================

HISTORY_FILE = "patient_history_retraining.json"

def load_patient_history():
    """Load patient prediction history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_patient_history(history):
    """Save patient prediction history to JSON file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_prediction_to_history(patient_id, patient_name, patient_data, prediction, confidence, risk_percentage):
    """Add new prediction to history with all features for retraining"""
    history = load_patient_history()

    # Store all 13 features for potential model retraining
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    patient_record = {
        'timestamp': datetime.now().isoformat(),
        'patient_id': patient_id,
        'patient_name': patient_name,
        'prediction': int(prediction),
        'risk_percentage': float(risk_percentage),
        'confidence': float(confidence),
        'target': int(prediction)  # The true label for retraining
    }

    # Add all 13 features
    for i, feature_name in enumerate(feature_names):
        patient_record[feature_name] = float(patient_data[0][i])

    history.append(patient_record)
    save_patient_history(history)

def prepare_training_data_from_history():
    """Prepare training data from patient history for retraining"""
    history = load_patient_history()
    if not history:
        return None, None

    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    # Extract features and targets from history
    X_new = []
    y_new = []

    for record in history:
        try:
            # Only use records that have all features
            features = [record.get(feature) for feature in feature_names]
            if all(f is not None for f in features):
                X_new.append(features)
                y_new.append(int(record.get('target', record.get('prediction', 0))))
        except:
            continue

    if len(X_new) < 5:  # Need minimum samples for training
        return None, None

    return np.array(X_new), np.array(y_new)

def retrain_model_with_new_data(original_data):
    """Retrain model combining original dataset with new patient data"""
    # Load original training data
    data = original_data
    X_original = data.drop('target', axis=1)
    y_original = data['target']

    # Get new data from history
    X_new, y_new = prepare_training_data_from_history()

    if X_new is None:
        return None, None, "Not enough patient data to retrain (need at least 5 records)"

    # Combine datasets
    X_combined = pd.concat([X_original, pd.DataFrame(X_new, columns=X_original.columns)], ignore_index=True)
    y_combined = np.concatenate([y_original, y_new])

    # Train and evaluate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    model = SVC(C=0.1, kernel='linear', gamma='scale', probability=True, random_state=42)

    # Use cross-validation for evaluation
    cv_scores = cross_val_score(model, X_scaled, y_combined, cv=5, scoring='accuracy')

    # Train final model
    model.fit(X_scaled, y_combined)

    return model, scaler, f"✅ Model retrained successfully!\n\nDataset: {len(X_original)} original + {len(X_new)} new patients = {len(X_combined)} total\nCV Accuracy: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})"

# ============================================================================
# MODEL TRAINING & PREDICTION
# ============================================================================

@st.cache_resource
def train_model():
    """Load data and train model once"""
    data = pd.read_csv('Heart_disease_cleveland_new.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM model
    model = SVC(C=0.1, kernel='linear', gamma='scale', probability=True, random_state=42)
    model.fit(X_scaled, y)

    return data, X, y, scaler, model

# Load model and data
data, X, y, scaler, model = train_model()

# ============================================================================
# SIDEBAR INPUT
# ============================================================================

st.sidebar.markdown("### 👤 Patient Information")
patient_name = st.sidebar.text_input("Patient Name (for history)", value="Anonymous")
patient_id = st.sidebar.text_input("Patient ID (e.g., P001, P002)", value="", placeholder="Unique ID to distinguish patients")

# Demographics section with background
st.sidebar.markdown("""
<div style='background-color: #E8F4F8; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #028090;'>
    <h3 style='color: #065A82; margin-top: 0;'>👥 Demographics</h3>
</div>
""", unsafe_allow_html=True)
age = st.sidebar.slider("Age (years)", min_value=29, max_value=77, value=55, step=1)
sex = st.sidebar.selectbox("Sex", ["Female (0)", "Male (1)"], index=1)
sex_val = int(sex.split("(")[1][0])

# Chest Symptoms section with background
st.sidebar.markdown("""
<div style='background-color: #FFF3E0; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #FF9800;'>
    <h3 style='color: #E65100; margin-top: 0;'>💓 Chest Symptoms</h3>
</div>
""", unsafe_allow_html=True)
cp_options = ["0 - Typical angina", "1 - Atypical angina", "2 - Non-anginal pain", "3 - Asymptomatic"]
cp_selected = st.sidebar.selectbox("Chest Pain Type", cp_options, index=0)
cp = int(cp_selected.split(" - ")[0])

# Blood Measurements section with background
st.sidebar.markdown("""
<div style='background-color: #F3E5F5; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #9C27B0;'>
    <h3 style='color: #6A1B9A; margin-top: 0;'>🩸 Blood Measurements</h3>
</div>
""", unsafe_allow_html=True)
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", min_value=94, max_value=192, value=130, step=1)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=126, max_value=353, value=240, step=5)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"], index=0)
fbs_val = int(fbs.split("(")[1][0])

# Exercise Stress Test section with background
st.sidebar.markdown("""
<div style='background-color: #E8F5E9; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;'>
    <h3 style='color: #1B5E20; margin-top: 0;'>🏃 Exercise Stress Test</h3>
</div>
""", unsafe_allow_html=True)
thalach = st.sidebar.slider("Max Heart Rate Achieved", min_value=71, max_value=202, value=150, step=1)
exang = st.sidebar.selectbox("Exercise-Induced Angina", ["No (0)", "Yes (1)"], index=0)
exang_val = int(exang.split("(")[1][0])
oldpeak = st.sidebar.slider("ST Depression (0-5.6)", min_value=0.0, max_value=5.6, value=1.0, step=0.1)

# ECG & Vessel Info section with background
st.sidebar.markdown("""
<div style='background-color: #FCE4EC; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #E91E63;'>
    <h3 style='color: #880E4F; margin-top: 0;'>⚡ ECG & Vessel Info</h3>
</div>
""", unsafe_allow_html=True)
restecg_options = ["0 - Normal", "1 - Abnormal", "2 - Severe"]
restecg_selected = st.sidebar.selectbox("Resting ECG Results", restecg_options, index=0)
restecg = int(restecg_selected.split(" - ")[0])

slope_options = ["1 - Upsloping", "2 - Flat", "3 - Downsloping"]
slope_selected = st.sidebar.selectbox("Slope of ST Segment", slope_options, index=0)
slope = int(slope_selected.split(" - ")[0])

ca = st.sidebar.slider("Major Vessels Colored (0-4)", min_value=0, max_value=4, value=0, step=1)

thal_options = ["0 - Normal", "1 - Fixed defect", "2 - Reversible defect", "3 - Normal variant"]
thal_selected = st.sidebar.selectbox("Thalassemia Type", thal_options, index=0)
thal = int(thal_selected.split(" - ")[0])

# ============================================================================
# PREDICTION
# ============================================================================

patient_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]])

# Use retrained model if available, otherwise use original
if st.session_state.use_retrained and st.session_state.retrained_model is not None:
    patient_data_scaled = st.session_state.retrained_scaler.transform(patient_data)
    prediction = st.session_state.retrained_model.predict(patient_data_scaled)[0]
    prediction_proba = st.session_state.retrained_model.predict_proba(patient_data_scaled)[0, 1]
else:
    patient_data_scaled = scaler.transform(patient_data)
    prediction = model.predict(patient_data_scaled)[0]
    prediction_proba = model.predict_proba(patient_data_scaled)[0, 1]

risk_percentage = prediction_proba * 100

# ============================================================================
# TOP METRICS (Original Style)
# ============================================================================


with tab1:
    # Top 3 Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color: #065A82; margin: 0;'>Risk Score</h3>
            <h2 style='color: #C0392B; margin: 0;'>{risk_percentage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        prediction_text = "⚠️ HIGH RISK" if prediction == 1 else "✅ LOW RISK"
        color = "#C0392B" if prediction == 1 else "#1E7E34"
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color: #065A82; margin: 0;'>Prediction</h3>
            <h2 style='color: {color}; margin: 0;'>{prediction_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        confidence = max(prediction_proba, 1-prediction_proba) * 100
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color: #065A82; margin: 0;'>Confidence</h3>
            <h2 style='color: #028090; margin: 0;'>{confidence:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        # Probability bar chart (original style)
        fig_prob = go.Figure(data=[
            go.Bar(
                y=['Disease', 'No Disease'],
                x=[prediction_proba * 100, (1 - prediction_proba) * 100],
                orientation='h',
                marker=dict(color=['#E57373', '#7BC67B']),
                text=[f"{prediction_proba*100:.1f}%", f"{(1-prediction_proba)*100:.1f}%"],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
            )
        ])
        fig_prob.update_layout(
            title="Prediction Probability",
            xaxis_title="Probability (%)",
            height=300,
            showlegend=False,
            template='plotly_white',
            margin=dict(l=100, r=50, t=50, b=30)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col2:
        # Risk gauge (original style)
        fig_gauge = go.Figure(data=[go.Indicator(
            mode="gauge+number+delta",
            value=risk_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#E57373"},
                'steps': [
                    {'range': [0, 30], 'color': "#7BC67B"},
                    {'range': [30, 70], 'color': "#FFD966"},
                    {'range': [70, 100], 'color': "#E57373"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}
            })])
        fig_gauge.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 SHAP Feature Importance")

    # SHAP Explainability
    with st.spinner("Calculating feature importance..."):
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(scaler.transform(x))[:, 1],
            shap.sample(X.values, min(100, len(X)))
        )

        # Get SHAP values for this patient
        shap_values = explainer.shap_values(patient_data_scaled)

    feature_names = X.columns.tolist()
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[0],
        'Absolute SHAP': np.abs(shap_values[0])
    }).sort_values('Absolute SHAP', ascending=False).head(8)

    # Visualize SHAP values
    fig_shap = go.Figure(data=[
        go.Bar(
            y=shap_importance['Feature'],
            x=shap_importance['SHAP Value'],
            orientation='h',
            marker=dict(
                color=shap_importance['SHAP Value'],
                colorscale='RdBu',
                cmid=0,
                showscale=False
            ),
            text=[f"{v:.4f}" for v in shap_importance['SHAP Value']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
        )
    ])

    fig_shap.update_layout(
        title="Top 8 Features Contributing to This Prediction",
        xaxis_title="SHAP Value (Red=Risk Increase, Blue=Risk Decrease)",
        yaxis_title="Feature",
        height=350,
        template='plotly_white',
        margin=dict(l=150, r=100, t=60, b=50)
    )

    st.plotly_chart(fig_shap, use_container_width=True)

    # Feature explanation details
    st.markdown("### Feature Impact Details")
    for idx, row in shap_importance.head(5).iterrows():
        feature = row['Feature']
        shap_val = row['SHAP Value']
        direction = '📈 Increases Risk' if shap_val > 0 else '📉 Decreases Risk'

        feature_idx = feature_names.index(feature)
        actual_value = patient_data[0][feature_idx]

        st.markdown(f"**{feature}** = {actual_value:.1f} {direction} (impact: {abs(shap_val):.4f})")

    # ============================================================================
    # CURRENT PREDICTION INPUT SUMMARY
    # ============================================================================

    st.markdown("---")
    st.subheader("📥 Current Prediction Input")
    st.caption("ℹ️ This shows the patient data you entered above via the sidebar. Compare with historical data in the Risk Trend Analysis section.")
    summary_df = pd.DataFrame({
        'Feature': ['Age', 'Sex', 'Chest Pain Type', 'Blood Pressure', 'Cholesterol',
                    'Blood Sugar', 'ECG Result', 'Max Heart Rate', 'Angina', 'ST Depression',
                    'ST Slope', 'Major Vessels', 'Thalassemia'],
        'Value': [age, 'Male' if sex_val == 1 else 'Female', cp, trestbps, chol,
                  'Yes' if fbs_val == 1 else 'No', restecg, thalach, 'Yes' if exang_val == 1 else 'No',
                  oldpeak, slope, ca, thal]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Save prediction button
    col_save, col_space = st.columns([1, 3])
    with col_save:
        if st.button("💾 Save to History", use_container_width=True):
            if patient_id.strip() == "":
                st.error("⚠️ Please enter a Patient ID before saving!")
            else:
                add_prediction_to_history(patient_id, patient_name, patient_data, prediction, max(prediction_proba, 1-prediction_proba), risk_percentage)
                st.success(f"✅ Prediction saved for {patient_name} (ID: {patient_id})! View it in the 'Patient History & Trends' tab.")

    st.markdown("---")

    # Information sections
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
                ### Model Details
                - **Algorithm:** Support Vector Machine (SVM)
                - **Kernel:** Linear
                - **Training Data:** 303 unique patients (Cleveland)
                - **Test Accuracy:** 80.3%
                - **ROC-AUC:** 0.947
                - **Recall:** 88% (catches disease cases)
                - **Enhancement:** Can be retrained with new patient data
                """)

    with col2:
        with st.expander("⚠️ Important Disclaimers"):
            st.markdown("""
                **EDUCATIONAL USE ONLY**

                - NOT for clinical diagnosis
                - Always consult healthcare professionals
                - Model trained on 1980s data
                - Predictions are estimates only
                """)

    with col3:
        with st.expander("📂 Source Code"):
            st.markdown("""
                ### GitHub Repository

                View the complete source code on GitHub:

                [![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-black?logo=github)](https://github.com/mahdi-20/heart-disease-predictor)

                **Features:**
                - ML model training
                - Streamlit web app
                - Patient history tracking
                - Model retraining
                - Risk analysis
                - SHAP explainability

                ### Two App Versions:
                - **app.py** - Standard version
                - **app_with_retraining.py** - With model retraining
                """)

    # Model Management Section
    with st.expander("🔧 Model Management & Retraining"):
        st.markdown("### Retrain Model with New Patient Data")
        st.caption("Combine original dataset with patient records saved in history to improve model performance")

        col_retrain, col_info = st.columns([2, 1])

        with col_retrain:
            if st.button("🔄 Retrain Model with New Data", use_container_width=True):
                with st.spinner("Retraining model with patient history data..."):
                    retrained_model, retrained_scaler, message = retrain_model_with_new_data(data)

                    if retrained_model is not None:
                        # Save the retrained model and scaler to session state
                        st.session_state.retrained_model = retrained_model
                        st.session_state.retrained_scaler = retrained_scaler
                        st.session_state.use_retrained = True
                        st.success(message)
                        st.balloons()
                    else:
                        st.warning(f"⚠️ {message}")

        with col_info:
            history_for_count = load_patient_history()
            st.metric("Patient Records", len(history_for_count))

        # Show which model is active
        st.markdown("---")
        if st.session_state.get('use_retrained', False):
            st.info("✅ **USING RETRAINED MODEL** - Model has been updated with new patient data")
        else:
            st.info("📊 **USING ORIGINAL MODEL** - No retraining performed yet")

# ============================================================================
# END OF TAB1 - START OF TAB2
# ============================================================================

with tab2:
    st.markdown("---")
    st.subheader("📋 Patient Prediction History")

    # Display history
    history = load_patient_history()

    if history:
        # Clear history button
        if st.button("🗑️ Clear All History", use_container_width=False):
            save_patient_history([])
            st.rerun()

        st.write(f"**Total Predictions Saved:** {len(history)}")

        history_df = pd.DataFrame(history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], format='ISO8601').dt.strftime('%Y-%m-%d %H:%M')

        display_cols = ['timestamp', 'patient_id', 'patient_name', 'age', 'chol', 'thalach', 'risk_percentage', 'prediction']
        display_df = history_df[display_cols].copy()
        display_df.columns = ['Date/Time', 'Patient ID', 'Patient Name', 'Age', 'Cholesterol', 'Max HR', 'Risk %', 'Status']
        display_df['Status'] = display_df['Status'].apply(lambda x: '🔴 Disease' if x == 1 else '🟢 No Disease')

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Trend analysis
        st.subheader("📈 Risk Trend Analysis")

        # Create patient labels with ID and name
        unique_patients = history_df[['patient_id', 'patient_name']].drop_duplicates().values
        patient_labels = [f"{pid} - {pname}" for pid, pname in unique_patients]
        patient_map = {label: (unique_patients[i][0], unique_patients[i][1]) for i, label in enumerate(patient_labels)}

        patient_selection_label = st.selectbox(
            "Select patient to view trends:",
            patient_labels
        )

        selected_id, selected_name = patient_map[patient_selection_label]
        patient_history_data = history_df[(history_df['patient_id'] == selected_id) & (history_df['patient_name'] == selected_name)].sort_values('timestamp')

        if len(patient_history_data) > 1:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=pd.to_datetime(patient_history_data['timestamp']),
                y=patient_history_data['risk_percentage'],
                mode='lines+markers',
                marker=dict(size=10, color='#FF6B6B'),
                line=dict(width=2, color='#FF6B6B'),
                hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>'
            ))

            fig_trend.update_layout(
                title=f"Risk Score Trend for {selected_name} (ID: {selected_id})",
                xaxis_title="Date/Time",
                yaxis_title="Risk Percentage (%)",
                height=300,
                template='plotly_white'
            )

            st.plotly_chart(fig_trend, use_container_width=True)

            # Display patient information below trend
            st.markdown("### Patient Information (From History)")
            latest_record = patient_history_data.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", f"{int(latest_record['age'])} years")
            with col2:
                st.metric("Cholesterol", f"{int(latest_record['chol'])} mg/dl")
            with col3:
                st.metric("Max Heart Rate", f"{int(latest_record['thalach'])} bpm")
            with col4:
                prediction_status = "🔴 Disease" if latest_record['prediction'] == 1 else "🟢 No Disease"
                st.metric("Latest Status", prediction_status)

            # Show all predictions for this patient
            st.markdown("### All Predictions for This Patient")
            display_cols = ['timestamp', 'risk_percentage', 'prediction', 'age', 'chol', 'thalach']
            patient_display_df = patient_history_data[display_cols].copy()
            patient_display_df.columns = ['Date/Time', 'Risk %', 'Status', 'Age', 'Cholesterol', 'Max HR']
            patient_display_df['Status'] = patient_display_df['Status'].apply(lambda x: '🔴 Disease' if x == 1 else '🟢 No Disease')
            st.dataframe(patient_display_df, use_container_width=True, hide_index=True)

        else:
            if len(patient_history_data) > 0:
                st.markdown("### 📊 Single Prediction - Need More Data for Trends")
                st.info(f"✅ {selected_name} (ID: {selected_id}) has **{len(patient_history_data)} prediction** on record. Save at least **2 predictions** to see trend analysis and risk patterns!")

                latest_record = patient_history_data.iloc[-1]

                # Show single prediction with metrics
                st.markdown("### Patient Information (Latest Prediction)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Age", f"{int(latest_record['age'])} years")
                with col2:
                    st.metric("Cholesterol", f"{int(latest_record['chol'])} mg/dl")
                with col3:
                    st.metric("Max Heart Rate", f"{int(latest_record['thalach'])} bpm")
                with col4:
                    prediction_status = "🔴 Disease" if latest_record['prediction'] == 1 else "🟢 No Disease"
                    st.metric("Status", prediction_status)

                # Show the single prediction details
                st.markdown("### Prediction Details")
                col1, col2 = st.columns(2)
                with col1:
                    risk_color = "🔴" if latest_record['prediction'] == 1 else "🟢"
                    st.metric("Risk Percentage", f"{latest_record['risk_percentage']:.1f}%")
                with col2:
                    st.metric("Confidence", f"{latest_record['confidence']*100:.1f}%")

                st.write(f"**Prediction Date**: {latest_record['timestamp']}")
                st.write("💡 **Tip**: Make more predictions for this patient to track trends over time!")
    else:
        st.info("No predictions saved yet. Save a prediction above to start tracking!")

# ============================================================================
# TAB 3 - ABOUT PROJECT
# ============================================================================

with tab3:
    st.markdown("<h2 style='text-align: center; color: #065A82;'>📚 About This Project</h2>", unsafe_allow_html=True)

    st.markdown("---")

    # Project Overview
    st.markdown("## 📋 Project Overview")
    st.markdown("""
    This is an **Advanced Machine Learning & Data Analytics** project that demonstrates the application
    of machine learning techniques to predict heart disease risk based on clinical features.

    The project showcases:
    - ✅ ML model development and evaluation
    - ✅ Interactive web application deployment
    - ✅ Data visualization and analysis
    - ✅ Model explainability (SHAP)
    - ✅ Continuous learning with model retraining
    - ✅ Patient history tracking and trend analysis
    """)

    st.markdown("---")

    # Course Information
    st.markdown("## 🎓 Course Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        ### Course
        **Advanced ML & Data Analytics**
        """)

    with col2:
        st.info("""
        ### Institution
        **Nexa-land**
        """)

    with col3:
        st.info("""
        ### Professor
        **Prof. Hamed Mamani**

        University of Washington
        """)

    st.markdown("---")

    # Author Information
    st.markdown("## 👨‍💻 Author")
    st.markdown("""
    **Mahdi Bakhtiari** (@mahdi-20)

    GitHub: [github.com/mahdi-20](https://github.com/mahdi-20)
    """)

    st.markdown("---")

    # Technologies Used
    st.markdown("## 🛠️ Technologies & Libraries")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Python ML Stack
        - **NumPy** - Numerical computing & arrays
        - **Pandas** - Data manipulation & analysis
        - **Scikit-Learn** - ML algorithms & preprocessing
        - **Matplotlib/Seaborn** - Data visualization
        - **Plotly** - Interactive visualizations

        ### ML Model Development
        - **SVM** - Support Vector Machine (Linear Kernel)
        - **Random Forest** - Ensemble classification
        - **Logistic Regression** - Baseline model
        - **Model Training & Evaluation**
        - **Cross-Validation** - 5-fold & 10-fold CV
        """)

    with col2:
        st.markdown("""
        ### ML Techniques & Concepts
        - **Preprocessing** - Feature scaling & normalization
        - **Feature Engineering** - Clinical feature selection
        - **Exploratory Data Analysis (EDA)** - Statistical analysis
        - **Classification Models** - Binary disease prediction
        - **Regression Concepts** - Model relationships
        - **Evaluation Metrics** - Accuracy, ROC-AUC, Recall, Precision

        ### Model Interpretability & Deployment
        - **SHAP** - Feature importance analysis
        - **Streamlit** - Interactive web framework
        - **Model Explainability** - Understanding predictions
        - **Interactive Visualizations** - Real-time insights
        """)

    st.markdown("---")

    # Dataset Information
    st.markdown("## 📊 Dataset")
    st.markdown("""
    - **Source:** Cleveland Heart Disease Dataset
    - **Samples:** 303 patients
    - **Features:** 13 clinical measurements
    - **Target:** Binary classification (presence/absence of heart disease)
    - **Train/Test Split:** Stratified k-fold cross-validation
    """)

    st.markdown("---")

    # Model Performance
    st.markdown("## 📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Algorithm", "SVM", "Linear Kernel")

    with col2:
        st.metric("Test Accuracy", "80.3%", "±2.3%")

    with col3:
        st.metric("ROC-AUC", "0.947", "High")

    with col4:
        st.metric("Recall", "88%", "Disease Detection")

    st.markdown("---")

    # Features
    st.markdown("## ✨ Application Features")
    st.markdown("""
    ✅ **Interactive Web Application Deployment** - Real-time predictions with Streamlit framework

    ✅ **Data Visualization and Analysis** - Interactive charts, gauges, and trend analysis

    ✅ **Model Explainability (SHAP)** - Understand which features most influence predictions

    ✅ **Patient History Tracking and Trend Analysis** - Save, compare, and analyze multiple predictions over time

    **Advanced Features:**
    - 🔄 Model Retraining - Improve model with new patient data (Advanced version only)
    - 👥 Patient ID Support for distinguishing similar patients
    - 🎯 Confidence Scores for each prediction
    - 📱 Responsive Design for desktop and mobile devices
    """)

    st.markdown("---")

    # Source Code
    st.markdown("## 📂 Source Code")
    st.markdown("""
    Full project source code available on GitHub:

    [![GitHub](https://img.shields.io/badge/GitHub-View%20Repository-black?logo=github&style=for-the-badge)](https://github.com/mahdi-20/heart-disease-predictor)

    **Repository includes:**
    - `app.py` - Standard version
    - `app_with_retraining.py` - Advanced version with model retraining
    - `Heart_disease_cleveland_new.csv` - Dataset
    - `requirements.txt` - Dependencies
    - Complete documentation
    """)

    st.markdown("---")

    # Disclaimer
    st.warning("""
    ### ⚠️ Important Disclaimer

    This application is for **educational purposes only** and should not be used for clinical diagnosis.
    Always consult with qualified healthcare professionals for medical advice and diagnosis.

    The model is trained on historical data and predictions are estimates only.
    """)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;'>
        <p>❤️ Built with Python, Machine Learning, and Streamlit</p>
        <p>Advanced ML & Data Analytics Course | Nexa-land | Prof. Hamed Mamani, University of Washington</p>
    </div>
    """, unsafe_allow_html=True)
