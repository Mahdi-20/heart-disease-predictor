"""
Heart Disease Prediction Model - Interactive Dash Dashboard
Comprehensive visualization of model performance, metrics, and feature importance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_curve, auc, classification_report,
                            accuracy_score, precision_score, recall_score, f1_score)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('Heart_disease_cleveland_new.csv')
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Heart Disease Prediction - Model Dashboard"

# Define custom color palette
colors = {
    'primary': '#065A82',
    'secondary': '#028090',
    'accent': '#00A896',
    'light': '#EBF4FA',
    'dark': '#0D3B4D',
    'success': '#1E7E34',
    'danger': '#C0392B',
    'warning': '#F39C12',
    'bg': '#F5F7FA'
}

# Train all models for comparison
models_dict = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'LDA': LinearDiscriminantAnalysis()
}

# Train SVM specifically
svm_model = SVC(C=0.1, kernel='linear', gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Train Random Forest for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)

# Get predictions for best model (SVM)
y_pred_test = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

# Cross-validation scores for all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model in models_dict.items():
    if name == 'SVM':
        model = SVC(kernel='linear', probability=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_scores[name] = scores.mean()

# Feature importance (from Random Forest)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# ============================================================================
# FIGURES
# ============================================================================

# 1. Model Comparison (Cross-validation)
fig_cv = go.Figure(data=[
    go.Bar(
        x=list(cv_scores.keys()),
        y=list(cv_scores.values()),
        marker=dict(color=[colors['secondary'], colors['accent'], colors['warning'], colors['primary']]),
        text=[f"{v:.1%}" for v in cv_scores.values()],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>CV Accuracy: %{y:.1%}<extra></extra>'
    )
])
fig_cv.update_layout(
    title='Model Comparison - 5-Fold Cross-Validation Accuracy',
    xaxis_title='Model',
    yaxis_title='Accuracy',
    yaxis=dict(range=[0, 1]),
    hovermode='x unified',
    template='plotly_white',
    height=400,
    showlegend=False,
    font=dict(family='Arial', size=12),
    margin=dict(l=50, r=50, t=60, b=50)
)

# 2. ROC Curve
fig_roc = go.Figure(data=[
    go.Scatter(x=fpr, y=tpr, mode='lines', name='SVM ROC',
               line=dict(color=colors['accent'], width=3),
               hovertemplate='False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra></extra>'),
    go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
               line=dict(color=colors['light'], width=2, dash='dash'),
               hovertemplate='False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra></extra>')
])
fig_roc.update_layout(
    title=f'ROC Curve (AUC = {roc_auc:.3f})',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    hovermode='closest',
    template='plotly_white',
    height=400,
    font=dict(family='Arial', size=12),
    margin=dict(l=50, r=50, t=60, b=50)
)

# 3. Confusion Matrix Heatmap
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicted No Disease', 'Predicted Disease'],
    y=['Actual No Disease', 'Actual Disease'],
    text=cm,
    texttemplate='%{text}',
    textfont={"size": 14},
    colorscale=[[0, colors['light']], [1, colors['primary']]],
    showscale=False,
    hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
))
fig_cm.update_layout(
    title='Confusion Matrix (Test Set)',
    xaxis_title='Predicted Label',
    yaxis_title='True Label',
    height=400,
    template='plotly_white',
    font=dict(family='Arial', size=12),
    margin=dict(l=50, r=50, t=60, b=50)
)

# 4. Feature Importance
fig_importance = go.Figure(data=[
    go.Bar(
        x=feature_importance['Importance'],
        y=feature_importance['Feature'],
        orientation='h',
        marker=dict(color=feature_importance['Importance'],
                   colorscale=[[0, colors['accent']], [1, colors['primary']]],
                   showscale=False),
        text=[f"{v:.1%}" for v in feature_importance['Importance']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.1%}<extra></extra>'
    )
])
fig_importance.update_layout(
    title='Top 10 Feature Importance (Random Forest)',
    xaxis_title='Importance Score',
    yaxis_title='Feature',
    height=400,
    template='plotly_white',
    showlegend=False,
    font=dict(family='Arial', size=12),
    margin=dict(l=150, r=50, t=60, b=50)
)

# 5. Classification Metrics
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [test_accuracy, test_precision, test_recall, test_f1]
}
metrics_df = pd.DataFrame(metrics_data)

fig_metrics = go.Figure(data=[
    go.Bar(
        x=metrics_df['Metric'],
        y=metrics_df['Score'],
        marker=dict(color=[colors['primary'], colors['secondary'], colors['accent'], colors['warning']]),
        text=[f"{v:.1%}" for v in metrics_df['Score']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1%}<extra></extra>'
    )
])
fig_metrics.update_layout(
    title='Classification Metrics (Test Set)',
    xaxis_title='Metric',
    yaxis_title='Score',
    yaxis=dict(range=[0, 1]),
    height=400,
    template='plotly_white',
    showlegend=False,
    font=dict(family='Arial', size=12),
    margin=dict(l=50, r=50, t=60, b=50)
)

# 6. Class Distribution
class_counts = y.value_counts().sort_index()
fig_class_dist = go.Figure(data=[
    go.Bar(
        x=['No Disease (0)', 'Disease (1)'],
        y=class_counts.values,
        marker=dict(color=[colors['success'], colors['danger']]),
        text=class_counts.values,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )
])
fig_class_dist.update_layout(
    title='Class Distribution (Full Dataset)',
    xaxis_title='Class',
    yaxis_title='Number of Samples',
    height=400,
    template='plotly_white',
    showlegend=False,
    font=dict(family='Arial', size=12),
    margin=dict(l=50, r=50, t=60, b=50)
)

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.H1('❤️ Heart Disease Prediction Model Dashboard',
                style={'color': colors['primary'], 'marginBottom': 10}),
        html.P('Comprehensive model evaluation, performance metrics, and feature analysis',
               style={'color': '#666', 'fontSize': 16})
    ], style={
        'backgroundColor': colors['light'],
        'padding': '30px',
        'marginBottom': '30px',
        'borderRadius': '10px',
        'borderLeft': f'5px solid {colors["accent"]}'
    }),

    # Metrics Row
    html.Div([
        html.Div([
            html.Div([
                html.H3('Test Accuracy', style={'color': colors['primary']}),
                html.H2(f'{test_accuracy:.1%}', style={'color': colors['secondary'], 'marginTop': 0})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className='metric-card', style={
            'flex': '1',
            'backgroundColor': colors['light'],
            'borderRadius': '10px',
            'borderLeft': f'4px solid {colors["primary"]}',
            'margin': '10px'
        }),

        html.Div([
            html.Div([
                html.H3('ROC-AUC', style={'color': colors['primary']}),
                html.H2(f'{roc_auc:.3f}', style={'color': colors['secondary'], 'marginTop': 0})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className='metric-card', style={
            'flex': '1',
            'backgroundColor': colors['light'],
            'borderRadius': '10px',
            'borderLeft': f'4px solid {colors["secondary"]}',
            'margin': '10px'
        }),

        html.Div([
            html.Div([
                html.H3('Recall (Disease)', style={'color': colors['primary']}),
                html.H2(f'{test_recall:.1%}', style={'color': colors['accent'], 'marginTop': 0})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className='metric-card', style={
            'flex': '1',
            'backgroundColor': colors['light'],
            'borderRadius': '10px',
            'borderLeft': f'4px solid {colors["accent"]}',
            'margin': '10px'
        }),

        html.Div([
            html.Div([
                html.H3('Precision', style={'color': colors['primary']}),
                html.H2(f'{test_precision:.1%}', style={'color': colors['warning'], 'marginTop': 0})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className='metric-card', style={
            'flex': '1',
            'backgroundColor': colors['light'],
            'borderRadius': '10px',
            'borderLeft': f'4px solid {colors["warning"]}',
            'margin': '10px'
        })
    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'marginBottom': '30px'
    }),

    # Visualizations Row 1
    html.Div([
        html.Div([
            dcc.Graph(figure=fig_cv)
        ], style={'flex': '1', 'minWidth': '45%', 'margin': '10px'}),

        html.Div([
            dcc.Graph(figure=fig_metrics)
        ], style={'flex': '1', 'minWidth': '45%', 'margin': '10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    # Visualizations Row 2
    html.Div([
        html.Div([
            dcc.Graph(figure=fig_roc)
        ], style={'flex': '1', 'minWidth': '45%', 'margin': '10px'}),

        html.Div([
            dcc.Graph(figure=fig_cm)
        ], style={'flex': '1', 'minWidth': '45%', 'margin': '10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    # Visualizations Row 3
    html.Div([
        html.Div([
            dcc.Graph(figure=fig_importance)
        ], style={'flex': '1', 'minWidth': '45%', 'margin': '10px'}),

        html.Div([
            dcc.Graph(figure=fig_class_dist)
        ], style={'flex': '1', 'minWidth': '45%', 'margin': '10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    # Footer
    html.Div([
        html.P('Built with Dash & Plotly | Dataset: Cleveland Heart Disease (303 unique patients) | Best Model: SVM (Tuned)',
               style={'color': '#666', 'textAlign': 'center', 'margin': 0})
    ], style={
        'backgroundColor': colors['light'],
        'padding': '20px',
        'marginTop': '30px',
        'borderRadius': '10px',
        'borderTop': f'2px solid {colors["accent"]}'
    })
], style={
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': colors['bg'],
    'padding': '20px',
    'maxWidth': '1400px',
    'margin': '0 auto'
})

if __name__ == '__main__':
    app.run(debug=True, port=8050)
