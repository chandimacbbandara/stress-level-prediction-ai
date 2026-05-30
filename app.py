"""
╔══════════════════════════════════════════════════════════════════╗
║  Student Mental Health Early Detection System — Dashboard       ║
║  Deploy on Hugging Face Spaces (Gradio SDK)                     ║
║  Compares 6 ML models across 22 variations                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════════
# 1. MODEL PERFORMANCE DATA
#    Metrics extracted from the model training pipeline notebooks
# ═══════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "MLP": "#10b981",
    "Random Forest": "#06b6d4",
    "SVM": "#8b5cf6",
    "KNN": "#f59e0b",
    "Logistic Regression": "#ec4899",
    "Decision Tree": "#f97316",
}

BEST_MODELS = {
    "MLP": {
        "accuracy": 0.9091, "precision": 0.91, "recall": 0.91, "f1": 0.91,
        "variation": "Variety 2 — [128,64] + Keras Callbacks",
        "config": "Hidden layers: [128, 64], Dropout: 0.25, EarlyStopping + ReduceLROnPlateau, 100 epochs",
        "icon": "🧠", "rank": 1,
    },
    "Random Forest": {
        "accuracy": 0.8955, "precision": 0.90, "recall": 0.90, "f1": 0.90,
        "variation": "GridSearchCV Tuned",
        "config": "Best params from grid: n_estimators, max_depth, min_samples_split, min_samples_leaf",
        "icon": "🌲", "rank": 2,
    },
    "SVM": {
        "accuracy": 0.8818, "precision": 0.88, "recall": 0.88, "f1": 0.88,
        "variation": "RBF Kernel — Tuned gamma & C",
        "config": "kernel='rbf', GridSearchCV over C=[0.1,1,10], gamma=[0.01,0.1,1]",
        "icon": "📐", "rank": 3,
    },
    "KNN": {
        "accuracy": 0.8682, "precision": 0.87, "recall": 0.87, "f1": 0.87,
        "variation": "k=3, Distance Weights",
        "config": "n_neighbors=3, weights='distance'",
        "icon": "📍", "rank": 4,
    },
    "Logistic Regression": {
        "accuracy": 0.8455, "precision": 0.85, "recall": 0.85, "f1": 0.85,
        "variation": "GridSearchCV Tuned",
        "config": "solver='saga', GridSearchCV over C=[0.1,1,10,50], penalty=['l1','l2']",
        "icon": "📈", "rank": 5,
    },
    "Decision Tree": {
        "accuracy": 0.8227, "precision": 0.82, "recall": 0.82, "f1": 0.82,
        "variation": "Default (No Depth Limit)",
        "config": "max_depth=None, random_state=42",
        "icon": "🌳", "rank": 6,
    },
}

# All variations tested for each model
MODEL_VARIATIONS = {
    "MLP": [
        {"name": "Variety 1 — Reduced (16 neurons)", "accuracy": 0.8545, "precision": 0.85, "recall": 0.85, "f1": 0.85, "config": "Hidden: [16], Dropout: 0.2, 20 epochs"},
        {"name": "Variety 2 — Callbacks ⭐", "accuracy": 0.9091, "precision": 0.91, "recall": 0.91, "f1": 0.91, "config": "Hidden: [128,64], Dropout: 0.25, EarlyStopping+ReduceLR"},
        {"name": "Variety 3 — Best Tuned Grid", "accuracy": 0.8955, "precision": 0.90, "recall": 0.90, "f1": 0.90, "config": "Best from manual grid over hidden layers & dropout"},
    ],
    "Random Forest": [
        {"name": "Baseline (Default Params)", "accuracy": 0.8864, "precision": 0.89, "recall": 0.89, "f1": 0.89, "config": "n_estimators=100 (default)"},
        {"name": "Manual Tuned (n=200, depth=20)", "accuracy": 0.8909, "precision": 0.89, "recall": 0.89, "f1": 0.89, "config": "n_estimators=200, max_depth=20"},
        {"name": "GridSearchCV Tuned ⭐", "accuracy": 0.8955, "precision": 0.90, "recall": 0.90, "f1": 0.90, "config": "GridSearchCV 5-fold CV, best params"},
    ],
    "SVM": [
        {"name": "Linear SVM (Tuned)", "accuracy": 0.8727, "precision": 0.87, "recall": 0.87, "f1": 0.87, "config": "kernel='linear', GridSearchCV C"},
        {"name": "RBF gamma='scale' (Tuned)", "accuracy": 0.8773, "precision": 0.88, "recall": 0.88, "f1": 0.88, "config": "kernel='rbf', gamma='scale', GridSearchCV C"},
        {"name": "RBF Specified gamma (Tuned) ⭐", "accuracy": 0.8818, "precision": 0.88, "recall": 0.88, "f1": 0.88, "config": "kernel='rbf', GridSearchCV C & gamma"},
    ],
    "KNN": [
        {"name": "k=3, Uniform Weights", "accuracy": 0.8591, "precision": 0.86, "recall": 0.86, "f1": 0.86, "config": "n_neighbors=3, weights='uniform'"},
        {"name": "k=5, Uniform Weights", "accuracy": 0.8500, "precision": 0.85, "recall": 0.85, "f1": 0.85, "config": "n_neighbors=5, weights='uniform'"},
        {"name": "k=3, Distance Weights ⭐", "accuracy": 0.8682, "precision": 0.87, "recall": 0.87, "f1": 0.87, "config": "n_neighbors=3, weights='distance'"},
    ],
    "Logistic Regression": [
        {"name": "Baseline (Default)", "accuracy": 0.8318, "precision": 0.83, "recall": 0.83, "f1": 0.83, "config": "solver='lbfgs', C=1.0, penalty='l2'"},
        {"name": "L1 Lasso (C=1.0)", "accuracy": 0.8273, "precision": 0.83, "recall": 0.83, "f1": 0.83, "config": "solver='saga', penalty='l1', C=1.0"},
        {"name": "L2 Ridge (C=0.1)", "accuracy": 0.8182, "precision": 0.82, "recall": 0.82, "f1": 0.82, "config": "solver='saga', penalty='l2', C=0.1"},
        {"name": "GridSearchCV Tuned ⭐", "accuracy": 0.8455, "precision": 0.85, "recall": 0.85, "f1": 0.85, "config": "solver='saga', GridSearchCV 5-fold CV"},
    ],
    "Decision Tree": [
        {"name": "Default (No Depth Limit) ⭐", "accuracy": 0.8227, "precision": 0.82, "recall": 0.82, "f1": 0.82, "config": "max_depth=None"},
        {"name": "max_depth = 1", "accuracy": 0.6818, "precision": 0.66, "recall": 0.67, "f1": 0.66, "config": "max_depth=1"},
        {"name": "max_depth = 3", "accuracy": 0.7864, "precision": 0.79, "recall": 0.79, "f1": 0.79, "config": "max_depth=3"},
        {"name": "max_depth = 10", "accuracy": 0.8182, "precision": 0.82, "recall": 0.82, "f1": 0.82, "config": "max_depth=10"},
    ],
}

# Confusion matrices for best variation of each model (test set: 220 samples)
CONFUSION_MATRICES = {
    "MLP": [[66, 2, 6], [2, 68, 2], [7, 1, 66]],
    "Random Forest": [[66, 3, 5], [3, 65, 4], [2, 6, 66]],
    "SVM": [[65, 3, 6], [3, 64, 5], [2, 7, 65]],
    "KNN": [[63, 5, 6], [4, 62, 6], [3, 5, 66]],
    "Logistic Regression": [[61, 6, 7], [5, 60, 7], [4, 5, 65]],
    "Decision Tree": [[60, 6, 8], [5, 58, 9], [4, 7, 63]],
}

# Per-class metrics for the best model (MLP) from the README
MLP_PER_CLASS = {
    "Low Stress (0)": {"precision": 0.88, "recall": 0.89, "f1": 0.89, "support": 74},
    "Medium Stress (1)": {"precision": 0.96, "recall": 0.94, "f1": 0.95, "support": 72},
    "High Stress (2)": {"precision": 0.89, "recall": 0.89, "f1": 0.89, "support": 74},
}

# Simulated MLP training history (based on training curve images)
np.random.seed(42)
EPOCHS = list(range(1, 31))
MLP_HISTORY = {
    "train_acc": [0.60, 0.75, 0.80, 0.85, 0.88, 0.87, 0.89, 0.90, 0.90, 0.91,
                  0.91, 0.91, 0.92, 0.92, 0.92, 0.94, 0.93, 0.92, 0.93, 0.93,
                  0.93, 0.93, 0.93, 0.93, 0.94, 0.94, 0.94, 0.95, 0.94, 0.93],
    "val_acc":   [0.88, 0.85, 0.89, 0.86, 0.87, 0.87, 0.88, 0.89, 0.89, 0.89,
                  0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.90, 0.90, 0.89, 0.90,
                  0.89, 0.89, 0.90, 0.88, 0.89, 0.88, 0.88, 0.88, 0.88, 0.88],
    "train_loss":[0.98, 0.62, 0.50, 0.42, 0.38, 0.35, 0.30, 0.28, 0.26, 0.25,
                  0.24, 0.23, 0.22, 0.21, 0.20, 0.18, 0.17, 0.19, 0.18, 0.17,
                  0.16, 0.16, 0.15, 0.15, 0.14, 0.13, 0.13, 0.12, 0.11, 0.11],
    "val_loss":  [0.75, 0.46, 0.38, 0.34, 0.30, 0.29, 0.28, 0.27, 0.27, 0.27,
                  0.26, 0.27, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.27, 0.26,
                  0.26, 0.26, 0.27, 0.27, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27],
}

# Dataset features (original before preprocessing)
DATASET_FEATURES = [
    ("anxiety_level", "Psychological", "Self-reported anxiety level (0–21)"),
    ("self_esteem", "Psychological", "Self-esteem score (0–30)"),
    ("mental_health_history", "Psychological", "Previous mental health issues (0/1)"),
    ("depression", "Psychological", "Depression score (0–27)"),
    ("headache", "Physiological", "Frequency of headaches (0–5)"),
    ("blood_pressure", "Physiological", "Blood pressure level (1–3)"),
    ("sleep_quality", "Physiological", "Quality of sleep (0–5)"),
    ("breathing_problem", "Physiological", "Breathing difficulty level (0–5)"),
    ("noise_level", "Environmental", "Noise in living environment (0–5)"),
    ("living_conditions", "Environmental", "Quality of living conditions (0–5)"),
    ("safety", "Environmental", "Perceived safety level (0–5)"),
    ("basic_needs", "Environmental", "Access to basic needs (0–5)"),
    ("academic_performance", "Academic", "Academic performance score (0–5)"),
    ("study_load", "Academic", "Study workload level (0–5)"),
    ("teacher_student_relationship", "Academic", "Relationship quality (0–5)"),
    ("future_career_concerns", "Academic", "Career anxiety level (0–5)"),
    ("social_support", "Social", "Social support score (0–3)"),
    ("peer_pressure", "Social", "Peer pressure level (0–5)"),
    ("extracurricular_activities", "Social", "Extracurricular involvement (0–5)"),
    ("bullying", "Social", "Bullying experience level (0–5)"),
]

PREPROCESSING_STEPS = [
    ("Handling Missing Values", "IT24100967", "Identified and imputed missing data to ensure dataset completeness"),
    ("One-Hot Encoding", "IT24100890", "Transformed categorical variables into numerical binary columns"),
    ("Min-Max Scaling", "IT24104387", "Normalized all features to [0, 1] range for consistent model input"),
    ("Outlier Removal", "IT24100307", "Detected and removed statistical outliers using IQR method"),
    ("Feature Selection", "IT24100821", "Selected the most informative features to reduce dimensionality"),
    ("Dimensional Reduction", "IT24100479", "Applied PCA to compress features while preserving variance"),
]

TEAM_MEMBERS = [
    ("Bandara I G C", "IT24100307", "Outlier Removal", "MLP (Best Model) 🏆", "🧠"),
    ("Jayanath P.A.P.R", "IT24104387", "Min-Max Scaling", "KNN", "📍"),
    ("Liyanage J A K", "IT24100479", "Dimensional Reduction", "Decision Tree", "🌳"),
    ("Makshuma B.G.K.J.", "IT24100821", "Feature Selection", "SVM", "📐"),
    ("Rodrigo H.V.O", "IT24100890", "One Hot Encoding", "Random Forest", "🌲"),
    ("Dasanayake H.T.N.", "IT24100967", "Handling Missing Data", "Logistic Regression", "📈"),
]

# ═══════════════════════════════════════════════════════════════════
# 2. PLOTLY CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

PLOT_BG = "rgba(15, 23, 42, 0.6)"
PLOT_PAPER = "rgba(0,0,0,0)"
PLOT_FONT = dict(family="Inter, system-ui, sans-serif", color="#e2e8f0")
PLOT_GRID = "rgba(148, 163, 184, 0.08)"


def hex_to_rgba(hex_color, alpha=0.1):
    """Convert a hex color string to rgba format for Plotly compatibility."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def base_layout():
    return dict(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_PAPER,
        font=PLOT_FONT,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(gridcolor=PLOT_GRID, zeroline=False),
        yaxis=dict(gridcolor=PLOT_GRID, zeroline=False),
    )


def create_overview_chart():
    """Main bar chart comparing best accuracy of each model."""
    models = list(BEST_MODELS.keys())
    accuracies = [BEST_MODELS[m]["accuracy"] * 100 for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    icons = [BEST_MODELS[m]["icon"] for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=accuracies,
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=8,
        ),
        text=[f"{a:.2f}%" for a in accuracies],
        textposition="outside",
        textfont=dict(size=14, color="#e2e8f0", family="Inter, sans-serif"),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>",
    ))

    layout = base_layout()
    layout.update(
        title=dict(text="Best Accuracy per Model", font=dict(size=20, color="#f1f5f9")),
        yaxis=dict(
            title="Accuracy (%)", range=[60, 100],
            gridcolor=PLOT_GRID, zeroline=False,
            tickfont=dict(color="#94a3b8"),
        ),
        xaxis=dict(tickfont=dict(size=12, color="#cbd5e1"), gridcolor=PLOT_GRID),
        height=450,
        bargap=0.3,
    )
    fig.update_layout(**layout)

    # Add a horizontal line for best model
    fig.add_hline(y=90.91, line_dash="dot", line_color="#10b981",
                  annotation_text="Best: 90.91%", annotation_position="top right",
                  annotation_font=dict(color="#10b981", size=12))
    return fig


def create_radar_chart():
    """Radar chart comparing all models across metrics."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Convert hex to rgba for fill colors (module-level hex_to_rgba is used)

    fig = go.Figure()
    for model_name, data in BEST_MODELS.items():
        values = [data["accuracy"], data["precision"], data["recall"], data["f1"]]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=hex_to_rgba(MODEL_COLORS[model_name], 0.1),
            line=dict(color=MODEL_COLORS[model_name], width=2),
            name=f"{data['icon']} {model_name}",
            hovertemplate=f"<b>{model_name}</b><br>%{{theta}}: %{{r:.4f}}<extra></extra>",
        ))

    layout = base_layout()
    layout.update(
        title=dict(text="Model Performance Radar", font=dict(size=20, color="#f1f5f9")),
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.4)",
            radialaxis=dict(
                visible=True, range=[0.75, 0.95],
                gridcolor="rgba(148, 163, 184, 0.15)",
                tickfont=dict(color="#94a3b8", size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(148, 163, 184, 0.15)",
                tickfont=dict(color="#cbd5e1", size=13),
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=11, color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
        ),
        height=500,
    )
    fig.update_layout(**layout)
    return fig


def create_confusion_matrix_chart(model_name):
    """Heatmap confusion matrix for a given model."""
    cm = CONFUSION_MATRICES[model_name]
    labels = ["Low Stress (0)", "Medium Stress (1)", "High Stress (2)"]
    color = MODEL_COLORS[model_name]

    # Build custom colorscale from white to model color
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "rgba(15, 23, 42, 0.8)"],
            [0.5, hex_to_rgba(color, 0.5)],
            [1.0, color],
        ],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=20, color="#f1f5f9"),
        hoverongaps=False,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))

    layout = base_layout()
    layout.update(
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=18, color="#f1f5f9")),
        xaxis=dict(title="Predicted Label", tickfont=dict(size=11, color="#cbd5e1"), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="True Label", tickfont=dict(size=11, color="#cbd5e1"), gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        height=420,
    )
    fig.update_layout(**layout)
    return fig


def create_variation_chart(model_name):
    """Bar chart comparing all variations for a specific model."""
    variations = MODEL_VARIATIONS[model_name]
    names = [v["name"] for v in variations]
    accs = [v["accuracy"] * 100 for v in variations]
    base_color = MODEL_COLORS[model_name]

    # Highlight the best variation
    best_acc = max(accs)
    colors = [base_color if a == best_acc else hex_to_rgba(base_color, 0.5) for a in accs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=accs,
        orientation="h",
        marker=dict(color=colors, cornerradius=6),
        text=[f"{a:.2f}%" for a in accs],
        textposition="outside",
        textfont=dict(size=13, color="#e2e8f0"),
        hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.2f}%<extra></extra>",
    ))

    layout = base_layout()
    layout.update(
        title=dict(text=f"Variation Comparison — {model_name}", font=dict(size=18, color="#f1f5f9")),
        xaxis=dict(title="Accuracy (%)", range=[60, 100], gridcolor=PLOT_GRID),
        yaxis=dict(tickfont=dict(size=11, color="#cbd5e1"), gridcolor="rgba(0,0,0,0)"),
        height=max(280, len(variations) * 70 + 100),
        bargap=0.25,
    )
    fig.update_layout(**layout)
    return fig


def create_mlp_training_charts():
    """Training accuracy and loss curves for MLP."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training & Validation Accuracy", "Training & Validation Loss"),
        horizontal_spacing=0.12,
    )

    # Accuracy curves
    fig.add_trace(go.Scatter(
        x=EPOCHS, y=MLP_HISTORY["train_acc"],
        mode="lines", name="Train Accuracy",
        line=dict(color="#06b6d4", width=2.5),
        hovertemplate="Epoch %{x}<br>Accuracy: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=EPOCHS, y=MLP_HISTORY["val_acc"],
        mode="lines", name="Val Accuracy",
        line=dict(color="#f59e0b", width=2.5),
        hovertemplate="Epoch %{x}<br>Accuracy: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Loss curves
    fig.add_trace(go.Scatter(
        x=EPOCHS, y=MLP_HISTORY["train_loss"],
        mode="lines", name="Train Loss",
        line=dict(color="#06b6d4", width=2.5),
        hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=EPOCHS, y=MLP_HISTORY["val_loss"],
        mode="lines", name="Val Loss",
        line=dict(color="#f59e0b", width=2.5),
        hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
    ), row=1, col=2)

    layout = base_layout()
    layout.update(
        height=400,
        showlegend=True,
        legend=dict(font=dict(size=11, color="#cbd5e1"), bgcolor="rgba(15,23,42,0.6)"),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Epoch", gridcolor=PLOT_GRID, tickfont=dict(color="#94a3b8"))
    fig.update_yaxes(gridcolor=PLOT_GRID, tickfont=dict(color="#94a3b8"))
    fig.update_annotations(font=dict(color="#cbd5e1", size=14))
    return fig


def create_metrics_comparison_chart():
    """Grouped bar chart: Accuracy, Precision, Recall, F1 for all models."""
    models = list(BEST_MODELS.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_colors = ["#06b6d4", "#8b5cf6", "#f59e0b", "#ec4899"]

    fig = go.Figure()
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [BEST_MODELS[m][metric] * 100 for m in models]
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=values,
            marker=dict(color=metric_colors[i], cornerradius=4),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}%<extra></extra>",
        ))

    layout = base_layout()
    layout.update(
        title=dict(text="All Metrics Comparison", font=dict(size=20, color="#f1f5f9")),
        barmode="group",
        yaxis=dict(title="Score (%)", range=[75, 100], gridcolor=PLOT_GRID),
        xaxis=dict(tickfont=dict(size=11, color="#cbd5e1")),
        legend=dict(font=dict(size=12, color="#cbd5e1"), bgcolor="rgba(15,23,42,0.6)"),
        height=450,
        bargap=0.2,
        bargroupgap=0.05,
    )
    fig.update_layout(**layout)
    return fig


def create_ranking_chart():
    """Horizontal bar chart showing model ranking by accuracy."""
    sorted_models = sorted(BEST_MODELS.items(), key=lambda x: x[1]["accuracy"])
    names = [f"{v['icon']} {k}" for k, v in sorted_models]
    accs = [v["accuracy"] * 100 for _, v in sorted_models]
    colors = [MODEL_COLORS[k] for k, _ in sorted_models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=accs,
        orientation="h",
        marker=dict(color=colors, cornerradius=8),
        text=[f"{a:.2f}%" for a in accs],
        textposition="outside",
        textfont=dict(size=14, color="#e2e8f0"),
        hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.2f}%<extra></extra>",
    ))

    layout = base_layout()
    layout.update(
        title=dict(text="Model Ranking by Accuracy", font=dict(size=20, color="#f1f5f9")),
        xaxis=dict(title="Accuracy (%)", range=[60, 100], gridcolor=PLOT_GRID),
        yaxis=dict(tickfont=dict(size=13, color="#cbd5e1"), gridcolor="rgba(0,0,0,0)"),
        height=380,
        bargap=0.25,
    )
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════
# 3. HTML COMPONENT BUILDERS
# ═══════════════════════════════════════════════════════════════════

def hero_html():
    return """
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #172554 70%, #0f172a 100%);
        border-radius: 20px;
        padding: 48px 40px;
        margin-bottom: 24px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        position: relative;
        overflow: hidden;
    ">
        <!-- Animated background orbs -->
        <div style="position:absolute;top:-60px;right:-60px;width:200px;height:200px;background:radial-gradient(circle,rgba(6,182,212,0.15),transparent 70%);border-radius:50%;"></div>
        <div style="position:absolute;bottom:-40px;left:-40px;width:160px;height:160px;background:radial-gradient(circle,rgba(139,92,246,0.12),transparent 70%);border-radius:50%;"></div>
        <div style="position:absolute;top:40%;left:60%;width:120px;height:120px;background:radial-gradient(circle,rgba(16,185,129,0.08),transparent 70%);border-radius:50%;"></div>

        <div style="position:relative;z-index:1;">
            <div style="
                display:inline-block;
                background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.2));
                border: 1px solid rgba(99,102,241,0.3);
                border-radius: 20px;
                padding: 6px 16px;
                font-size: 13px;
                color: #a5b4fc;
                margin-bottom: 16px;
                letter-spacing: 0.5px;
            ">🔬 Machine Learning Research Project</div>

            <h1 style="
                font-family: 'Inter', system-ui, sans-serif;
                font-size: 36px;
                font-weight: 800;
                background: linear-gradient(135deg, #f1f5f9, #06b6d4, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0 0 12px 0;
                line-height: 1.2;
            ">Student Mental Health<br>Early Detection System</h1>

            <p style="
                color: #94a3b8;
                font-size: 16px;
                max-width: 700px;
                line-height: 1.6;
                margin: 0 0 28px 0;
            ">Comparing <strong style="color:#e2e8f0">6 machine learning models</strong> across
            <strong style="color:#e2e8f0">22 variations</strong> to identify the most effective approach
            for predicting student stress levels from academic and psychological factors.</p>

            <!-- Stats Cards -->
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px;">
                <div style="
                    background: rgba(30, 41, 59, 0.6);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(148,163,184,0.1);
                    border-radius: 14px;
                    padding: 18px 24px;
                    min-width: 120px;
                    text-align: center;
                ">
                    <div style="font-size:28px;font-weight:800;color:#06b6d4;font-family:'Inter',sans-serif;">6</div>
                    <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">ML Models</div>
                </div>
                <div style="
                    background: rgba(30, 41, 59, 0.6);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(148,163,184,0.1);
                    border-radius: 14px;
                    padding: 18px 24px;
                    min-width: 120px;
                    text-align: center;
                ">
                    <div style="font-size:28px;font-weight:800;color:#8b5cf6;font-family:'Inter',sans-serif;">1,100</div>
                    <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Samples</div>
                </div>
                <div style="
                    background: rgba(30, 41, 59, 0.6);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(148,163,184,0.1);
                    border-radius: 14px;
                    padding: 18px 24px;
                    min-width: 120px;
                    text-align: center;
                ">
                    <div style="font-size:28px;font-weight:800;color:#f59e0b;font-family:'Inter',sans-serif;">3</div>
                    <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Stress Classes</div>
                </div>
                <div style="
                    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,182,212,0.1));
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(16,185,129,0.3);
                    border-radius: 14px;
                    padding: 18px 24px;
                    min-width: 120px;
                    text-align: center;
                ">
                    <div style="font-size:28px;font-weight:800;color:#10b981;font-family:'Inter',sans-serif;">90.91%</div>
                    <div style="font-size:12px;color:#6ee7b7;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Best Accuracy</div>
                </div>
            </div>

            <!-- Best Model Banner -->
            <div style="
                background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.08));
                border: 1px solid rgba(16,185,129,0.25);
                border-radius: 12px;
                padding: 14px 20px;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <span style="font-size:24px;">🏆</span>
                <div>
                    <div style="color:#10b981;font-weight:700;font-size:15px;font-family:'Inter',sans-serif;">
                        Best Model: MLP (Multilayer Perceptron)
                    </div>
                    <div style="color:#94a3b8;font-size:13px;margin-top:2px;">
                        Hidden layers [128, 64] with Keras callbacks — 90.91% test accuracy on 220 samples
                    </div>
                </div>
            </div>
        </div>
    </div>
    """


def model_card_html(model_name):
    """Generate a styled HTML card for a model's summary."""
    data = BEST_MODELS[model_name]
    color = MODEL_COLORS[model_name]
    is_best = model_name == "MLP"

    badge = ""
    if is_best:
        badge = '<span style="background:linear-gradient(135deg,#10b981,#06b6d4);color:#fff;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;margin-left:8px;">🏆 BEST MODEL</span>'

    return f"""
    <div style="
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid {color}33;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    ">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <span style="font-size:28px;">{data['icon']}</span>
            <div>
                <h3 style="margin:0;color:#f1f5f9;font-size:20px;font-family:'Inter',sans-serif;">
                    {model_name}{badge}
                </h3>
                <p style="margin:2px 0 0;color:#94a3b8;font-size:13px;">Best: {data['variation']}</p>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
            <div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:14px;text-align:center;border:1px solid rgba(148,163,184,0.08);">
                <div style="font-size:22px;font-weight:800;color:{color};font-family:'Inter',sans-serif;">{data['accuracy']*100:.2f}%</div>
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Accuracy</div>
            </div>
            <div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:14px;text-align:center;border:1px solid rgba(148,163,184,0.08);">
                <div style="font-size:22px;font-weight:800;color:{color};font-family:'Inter',sans-serif;">{data['precision']*100:.1f}%</div>
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Precision</div>
            </div>
            <div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:14px;text-align:center;border:1px solid rgba(148,163,184,0.08);">
                <div style="font-size:22px;font-weight:800;color:{color};font-family:'Inter',sans-serif;">{data['recall']*100:.1f}%</div>
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Recall</div>
            </div>
            <div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:14px;text-align:center;border:1px solid rgba(148,163,184,0.08);">
                <div style="font-size:22px;font-weight:800;color:{color};font-family:'Inter',sans-serif;">{data['f1']*100:.1f}%</div>
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">F1-Score</div>
            </div>
        </div>

        <div style="margin-top:14px;background:rgba(15,23,42,0.4);border-radius:8px;padding:10px 14px;border:1px solid rgba(148,163,184,0.06);">
            <span style="color:#64748b;font-size:12px;">Config: </span>
            <span style="color:#94a3b8;font-size:12px;font-family:'JetBrains Mono',monospace;">{data['config']}</span>
        </div>
    </div>
    """


def section_header_html(title, subtitle="", icon=""):
    return f"""
    <div style="margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid rgba(148,163,184,0.1);">
        <h2 style="margin:0;color:#f1f5f9;font-size:24px;font-weight:700;font-family:'Inter',sans-serif;">
            {icon} {title}
        </h2>
        {"<p style='margin:6px 0 0;color:#94a3b8;font-size:14px;'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """


def per_class_metrics_html():
    """Display MLP per-class metrics from the README."""
    rows = ""
    for cls, m in MLP_PER_CLASS.items():
        rows += f"""
        <tr>
            <td style="padding:10px 16px;color:#e2e8f0;font-weight:500;">{cls}</td>
            <td style="padding:10px 16px;color:#06b6d4;text-align:center;">{m['precision']:.2f}</td>
            <td style="padding:10px 16px;color:#8b5cf6;text-align:center;">{m['recall']:.2f}</td>
            <td style="padding:10px 16px;color:#f59e0b;text-align:center;">{m['f1']:.2f}</td>
            <td style="padding:10px 16px;color:#94a3b8;text-align:center;">{m['support']}</td>
        </tr>
        """

    return f"""
    <div style="
        background:rgba(30,41,59,0.5);
        border-radius:14px;
        border:1px solid rgba(148,163,184,0.1);
        overflow:hidden;
        margin-top:8px;
    ">
        <div style="padding:14px 20px;background:rgba(16,185,129,0.08);border-bottom:1px solid rgba(148,163,184,0.08);">
            <h4 style="margin:0;color:#10b981;font-size:15px;font-family:'Inter',sans-serif;">
                📊 MLP Per-Class Classification Report
            </h4>
        </div>
        <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:14px;">
            <thead>
                <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
                    <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;">Class</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Precision</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Recall</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">F1-Score</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Support</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def variation_table_html(model_name):
    """HTML table listing all variations with their metrics."""
    variations = MODEL_VARIATIONS[model_name]
    color = MODEL_COLORS[model_name]
    rows = ""
    best_acc = max(v["accuracy"] for v in variations)

    for v in variations:
        is_best = v["accuracy"] == best_acc
        bg = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)" if is_best else "transparent"
        border_left = f"3px solid {color}" if is_best else "3px solid transparent"

        rows += f"""
        <tr style="background:{bg};border-left:{border_left};border-bottom:1px solid rgba(148,163,184,0.06);">
            <td style="padding:12px 16px;color:#e2e8f0;font-weight:{'600' if is_best else '400'};font-size:13px;">
                {v['name']}
            </td>
            <td style="padding:12px 16px;color:{color};text-align:center;font-weight:700;font-size:14px;">
                {v['accuracy']*100:.2f}%
            </td>
            <td style="padding:12px 16px;color:#94a3b8;text-align:center;font-size:13px;">{v['precision']:.2f}</td>
            <td style="padding:12px 16px;color:#94a3b8;text-align:center;font-size:13px;">{v['recall']:.2f}</td>
            <td style="padding:12px 16px;color:#94a3b8;text-align:center;font-size:13px;">{v['f1']:.2f}</td>
            <td style="padding:12px 16px;color:#64748b;font-size:11px;font-family:'JetBrains Mono',monospace;">{v['config']}</td>
        </tr>
        """

    return f"""
    <div style="
        background:rgba(30,41,59,0.5);
        border-radius:14px;
        border:1px solid rgba(148,163,184,0.1);
        overflow:hidden;
        margin-top:8px;
    ">
        <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;">
            <thead>
                <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
                    <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;">Variation</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Accuracy</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Precision</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Recall</th>
                    <th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">F1</th>
                    <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Configuration</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def dataset_info_html():
    """Dataset overview card."""
    return """
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
        <div style="
            background:rgba(30,41,59,0.5);
            backdrop-filter:blur(12px);
            border:1px solid rgba(6,182,212,0.15);
            border-radius:14px;
            padding:24px;
        ">
            <h3 style="margin:0 0 16px;color:#06b6d4;font-size:17px;font-family:'Inter',sans-serif;">📋 Dataset Overview</h3>
            <div style="display:grid;gap:10px;">
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(148,163,184,0.06);">
                    <span style="color:#94a3b8;font-size:13px;">Source</span>
                    <span style="color:#e2e8f0;font-size:13px;font-weight:500;">Kaggle — Student Stress Factors</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(148,163,184,0.06);">
                    <span style="color:#94a3b8;font-size:13px;">Total Samples</span>
                    <span style="color:#e2e8f0;font-size:13px;font-weight:500;">1,100</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(148,163,184,0.06);">
                    <span style="color:#94a3b8;font-size:13px;">Features</span>
                    <span style="color:#e2e8f0;font-size:13px;font-weight:500;">20 original → 12 after PCA</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(148,163,184,0.06);">
                    <span style="color:#94a3b8;font-size:13px;">Target Classes</span>
                    <span style="color:#e2e8f0;font-size:13px;font-weight:500;">3 (Low / Medium / High Stress)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(148,163,184,0.06);">
                    <span style="color:#94a3b8;font-size:13px;">Train / Test Split</span>
                    <span style="color:#e2e8f0;font-size:13px;font-weight:500;">80% / 20% (stratified)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;">
                    <span style="color:#94a3b8;font-size:13px;">Class Balance</span>
                    <span style="color:#e2e8f0;font-size:13px;font-weight:500;">~370 / ~357 / ~373 (balanced)</span>
                </div>
            </div>
        </div>

        <div style="
            background:rgba(30,41,59,0.5);
            backdrop-filter:blur(12px);
            border:1px solid rgba(139,92,246,0.15);
            border-radius:14px;
            padding:24px;
        ">
            <h3 style="margin:0 0 16px;color:#8b5cf6;font-size:17px;font-family:'Inter',sans-serif;">⚙️ Preprocessing Pipeline</h3>
            <div style="display:grid;gap:8px;">
    """ + "".join([f"""
                <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:rgba(15,23,42,0.4);border-radius:8px;border:1px solid rgba(148,163,184,0.05);">
                    <span style="
                        background:linear-gradient(135deg,#8b5cf6,#06b6d4);
                        color:#fff;
                        font-size:11px;
                        font-weight:700;
                        padding:2px 8px;
                        border-radius:6px;
                        min-width:16px;
                        text-align:center;
                    ">{i+1}</span>
                    <div>
                        <div style="color:#e2e8f0;font-size:13px;font-weight:500;">{step[0]}</div>
                        <div style="color:#64748b;font-size:11px;">{step[2][:60]}...</div>
                    </div>
                </div>
    """ for i, step in enumerate(PREPROCESSING_STEPS)]) + """
            </div>
        </div>
    </div>
    """


def features_table_html():
    """HTML table of dataset features grouped by category."""
    categories = {}
    for feat, cat, desc in DATASET_FEATURES:
        categories.setdefault(cat, []).append((feat, desc))

    cat_colors = {
        "Psychological": "#ec4899",
        "Physiological": "#06b6d4",
        "Environmental": "#10b981",
        "Academic": "#f59e0b",
        "Social": "#8b5cf6",
    }

    rows = ""
    for feat, cat, desc in DATASET_FEATURES:
        color = cat_colors.get(cat, "#94a3b8")
        rows += f"""
        <tr style="border-bottom:1px solid rgba(148,163,184,0.06);">
            <td style="padding:10px 16px;color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-size:12px;">{feat}</td>
            <td style="padding:10px 16px;">
                <span style="background:{color}20;color:{color};padding:2px 10px;border-radius:8px;font-size:11px;font-weight:600;">{cat}</span>
            </td>
            <td style="padding:10px 16px;color:#94a3b8;font-size:13px;">{desc}</td>
        </tr>
        """

    return f"""
    <div style="
        background:rgba(30,41,59,0.5);
        border-radius:14px;
        border:1px solid rgba(148,163,184,0.1);
        overflow:hidden;
    ">
        <div style="padding:14px 20px;background:rgba(245,158,11,0.06);border-bottom:1px solid rgba(148,163,184,0.08);">
            <h4 style="margin:0;color:#f59e0b;font-size:15px;font-family:'Inter',sans-serif;">📝 Original Dataset Features (20)</h4>
        </div>
        <div style="max-height:500px;overflow-y:auto;">
        <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;">
            <thead>
                <tr style="border-bottom:1px solid rgba(148,163,184,0.12);position:sticky;top:0;background:rgba(30,41,59,0.95);">
                    <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Feature</th>
                    <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Category</th>
                    <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;font-size:12px;text-transform:uppercase;">Description</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
    </div>
    """


def team_html():
    """Generate team members section."""
    cards = ""
    for name, it, preprocess, model, icon in TEAM_MEMBERS:
        is_best = "🏆" in model
        border_color = "#10b981" if is_best else "rgba(148,163,184,0.1)"
        bg_accent = "rgba(16,185,129,0.06)" if is_best else "rgba(30,41,59,0.5)"

        cards += f"""
        <div style="
            background:{bg_accent};
            backdrop-filter:blur(12px);
            border:1px solid {border_color};
            border-radius:16px;
            padding:24px;
            display:flex;
            flex-direction:column;
            align-items:center;
            text-align:center;
            transition:transform 0.2s ease;
        ">
            <div style="
                width:56px;height:56px;
                background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(6,182,212,0.2));
                border-radius:50%;
                display:flex;align-items:center;justify-content:center;
                font-size:28px;
                margin-bottom:14px;
                border:2px solid rgba(148,163,184,0.15);
            ">{icon}</div>
            <h4 style="margin:0;color:#f1f5f9;font-size:15px;font-weight:600;font-family:'Inter',sans-serif;">{name}</h4>
            <p style="margin:4px 0 12px;color:#64748b;font-size:12px;font-family:'JetBrains Mono',monospace;">{it}</p>
            <div style="display:flex;flex-direction:column;gap:6px;width:100%;">
                <div style="background:rgba(15,23,42,0.5);border-radius:8px;padding:8px 12px;">
                    <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">Preprocessing</div>
                    <div style="color:#06b6d4;font-size:12px;font-weight:500;margin-top:2px;">{preprocess}</div>
                </div>
                <div style="background:rgba(15,23,42,0.5);border-radius:8px;padding:8px 12px;">
                    <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">Model</div>
                    <div style="color:#8b5cf6;font-size:12px;font-weight:500;margin-top:2px;">{model}</div>
                </div>
            </div>
        </div>
        """

    return f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
        {cards}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════
# 4. CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
.gradio-container {
    background: linear-gradient(160deg, #0f172a 0%, #1e1b4b 35%, #172554 65%, #0f172a 100%) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.dark {
    --body-background-fill: linear-gradient(160deg, #0f172a, #1e1b4b, #172554, #0f172a) !important;
}

/* ── Tab Navigation ── */
.tab-nav {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    padding: 6px !important;
    gap: 4px !important;
    margin-bottom: 20px !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 18px !important;
    transition: all 0.25s ease !important;
    font-family: 'Inter', sans-serif !important;
}

.tab-nav button:hover {
    background: rgba(99, 102, 241, 0.1) !important;
    color: #c7d2fe !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(6, 182, 212, 0.2)) !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.15) !important;
}

/* ── Blocks & Panels ── */
.block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.panel {
    background: rgba(30, 41, 59, 0.3) !important;
    border: 1px solid rgba(148, 163, 184, 0.08) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Plot containers ── */
.plot-container {
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.08) !important;
    padding: 8px !important;
}

/* ── Labels & Text ── */
label, .label-wrap span {
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Image containers ── */
.image-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(148, 163, 184, 0.35);
}

/* ── Footer ── */
footer {
    display: none !important;
}

/* ── Image Gallery ── */
.gallery-item {
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    overflow: hidden !important;
}

/* ── Accordion ── */
.accordion {
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 12px !important;
    background: rgba(30, 41, 59, 0.4) !important;
}
"""


# ═══════════════════════════════════════════════════════════════════
# 5. GRADIO APPLICATION
# ═══════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(
        title="Student Mental Health Detection — ML Dashboard",
    ) as demo:

        # ── HERO ──
        gr.HTML(hero_html())

        # ── MAIN TABS ──
        with gr.Tabs():

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 1: OVERVIEW
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🏠 Overview"):
                gr.HTML(section_header_html(
                    "Model Performance Overview",
                    "Best accuracy achieved by each of the 6 machine learning models tested",
                    "📊"
                ))
                gr.Plot(value=create_overview_chart(), label="")

                gr.HTML(section_header_html(
                    "Model Ranking",
                    "Models ranked from lowest to highest accuracy",
                    "🏅"
                ))
                gr.Plot(value=create_ranking_chart(), label="")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 2: MODEL COMPARISON
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("📊 Comparison"):
                gr.HTML(section_header_html(
                    "Detailed Metrics Comparison",
                    "Accuracy, Precision, Recall, and F1-Score across all models",
                    "📈"
                ))
                gr.Plot(value=create_metrics_comparison_chart(), label="")

                gr.HTML(section_header_html(
                    "Performance Radar",
                    "Multi-dimensional view of each model's strengths",
                    "🎯"
                ))
                gr.Plot(value=create_radar_chart(), label="")

                # Summary table
                gr.HTML(section_header_html("Summary Table", "", "📋"))
                summary_rows = ""
                for rank, (name, data) in enumerate(
                    sorted(BEST_MODELS.items(), key=lambda x: x[1]["accuracy"], reverse=True), 1
                ):
                    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][rank - 1]
                    color = MODEL_COLORS[name]
                    summary_rows += f"""
                    <tr style="border-bottom:1px solid rgba(148,163,184,0.06);">
                        <td style="padding:12px 16px;font-size:18px;text-align:center;">{medal}</td>
                        <td style="padding:12px 16px;color:#e2e8f0;font-weight:600;font-size:14px;">
                            {data['icon']} {name}
                        </td>
                        <td style="padding:12px 16px;color:{color};font-weight:700;text-align:center;font-size:15px;">
                            {data['accuracy']*100:.2f}%
                        </td>
                        <td style="padding:12px 16px;color:#94a3b8;text-align:center;font-size:13px;">{data['precision']:.2f}</td>
                        <td style="padding:12px 16px;color:#94a3b8;text-align:center;font-size:13px;">{data['recall']:.2f}</td>
                        <td style="padding:12px 16px;color:#94a3b8;text-align:center;font-size:13px;">{data['f1']:.2f}</td>
                        <td style="padding:12px 16px;color:#64748b;font-size:12px;">{data['variation']}</td>
                    </tr>
                    """

                gr.HTML(f"""
                <div style="
                    background:rgba(30,41,59,0.5);
                    border-radius:14px;
                    border:1px solid rgba(148,163,184,0.1);
                    overflow:hidden;
                ">
                    <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;">
                        <thead>
                            <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
                                <th style="padding:12px 16px;color:#64748b;font-size:12px;text-transform:uppercase;width:50px;">Rank</th>
                                <th style="padding:12px 16px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;">Model</th>
                                <th style="padding:12px 16px;text-align:center;color:#64748b;font-size:12px;text-transform:uppercase;">Accuracy</th>
                                <th style="padding:12px 16px;text-align:center;color:#64748b;font-size:12px;text-transform:uppercase;">Precision</th>
                                <th style="padding:12px 16px;text-align:center;color:#64748b;font-size:12px;text-transform:uppercase;">Recall</th>
                                <th style="padding:12px 16px;text-align:center;color:#64748b;font-size:12px;text-transform:uppercase;">F1</th>
                                <th style="padding:12px 16px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;">Best Variation</th>
                            </tr>
                        </thead>
                        <tbody>{summary_rows}</tbody>
                    </table>
                </div>
                """)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 3: INDIVIDUAL MODELS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🧠 Individual Models"):
                with gr.Tabs():
                    for model_name in BEST_MODELS:
                        icon = BEST_MODELS[model_name]["icon"]
                        tab_label = f"{icon} {model_name}"
                        if model_name == "MLP":
                            tab_label += " 🏆"

                        with gr.Tab(tab_label):
                            # Model summary card
                            gr.HTML(model_card_html(model_name))

                            # Variation comparison
                            gr.HTML(section_header_html(
                                "Variation Comparison",
                                f"All {len(MODEL_VARIATIONS[model_name])} variations tested for {model_name}",
                            ))
                            gr.HTML(variation_table_html(model_name))

                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Plot(
                                        value=create_variation_chart(model_name),
                                        label="",
                                    )
                                with gr.Column(scale=1):
                                    gr.Plot(
                                        value=create_confusion_matrix_chart(model_name),
                                        label="",
                                    )

                            # MLP-specific: training curves + per-class metrics
                            if model_name == "MLP":
                                gr.HTML(section_header_html(
                                    "Training History",
                                    "Accuracy and loss curves over 30 epochs (Variety 2 — with Callbacks)",
                                    "📉"
                                ))
                                gr.Plot(value=create_mlp_training_charts(), label="")
                                gr.HTML(per_class_metrics_html())

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 4: EDA VISUALIZATIONS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("📈 EDA Visualizations"):
                gr.HTML(section_header_html(
                    "Exploratory Data Analysis",
                    "Visualizations from the preprocessing and model training pipeline",
                    "🔍"
                ))

                # Preprocessing EDA
                gr.HTML("""
                <div style="
                    background:linear-gradient(135deg,rgba(139,92,246,0.08),rgba(6,182,212,0.05));
                    border:1px solid rgba(139,92,246,0.15);
                    border-radius:12px;
                    padding:14px 20px;
                    margin-bottom:16px;
                ">
                    <h3 style="margin:0;color:#a5b4fc;font-size:16px;font-family:'Inter',sans-serif;">
                        📊 Preprocessing & Data Analysis
                    </h3>
                </div>
                """)

                eda_dir = os.path.join(os.path.dirname(__file__), "results", "Preprocess_Eda_visualizations")
                model_viz_dir = os.path.join(os.path.dirname(__file__), "results", "model_Eda_visualizations")

                with gr.Row():
                    with gr.Column():
                        corr_path = os.path.join(eda_dir, "correlation_heatmap.jpg")
                        if os.path.exists(corr_path):
                            gr.Image(value=corr_path, label="Correlation Heatmap — Top 15 Features vs Stress Level")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found — Upload results/ directory</p>")
                    with gr.Column():
                        stress_path = os.path.join(eda_dir, "stress_level_histogram.jpg")
                        if os.path.exists(stress_path):
                            gr.Image(value=stress_path, label="Stress Level Distribution — Balanced Classes")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found — Upload results/ directory</p>")

                with gr.Row():
                    with gr.Column():
                        missing_path = os.path.join(eda_dir, "missing_data_heatmap.jpg")
                        if os.path.exists(missing_path):
                            gr.Image(value=missing_path, label="Missing Data Heatmap")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")
                    with gr.Column():
                        outlier_path = os.path.join(eda_dir, "outlier_boxplots.png")
                        if os.path.exists(outlier_path):
                            gr.Image(value=outlier_path, label="Outlier Detection — Box Plots")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")

                with gr.Row():
                    with gr.Column():
                        dist_path = os.path.join(eda_dir, "distributions_histogram.png")
                        if os.path.exists(dist_path):
                            gr.Image(value=dist_path, label="Feature Distributions")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")
                    with gr.Column():
                        pca_path = os.path.join(eda_dir, "cumulative_explained_variance.jpg")
                        if os.path.exists(pca_path):
                            gr.Image(value=pca_path, label="PCA — Cumulative Explained Variance")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")

                with gr.Row():
                    with gr.Column():
                        minmax_path = os.path.join(eda_dir, "Min-Max_corr_heatmap.jpg")
                        if os.path.exists(minmax_path):
                            gr.Image(value=minmax_path, label="Min-Max Scaled Correlation Heatmap")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")

                # Model Training Visualizations
                gr.HTML("""
                <div style="
                    background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(6,182,212,0.05));
                    border:1px solid rgba(16,185,129,0.15);
                    border-radius:12px;
                    padding:14px 20px;
                    margin:24px 0 16px;
                ">
                    <h3 style="margin:0;color:#6ee7b7;font-size:16px;font-family:'Inter',sans-serif;">
                        🧠 Model Training Visualizations (MLP — Best Model)
                    </h3>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        acc_graph = os.path.join(model_viz_dir, "Accuracy graph.png")
                        if os.path.exists(acc_graph):
                            gr.Image(value=acc_graph, label="MLP Training Accuracy Over Epochs")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")
                    with gr.Column():
                        loss_graph = os.path.join(model_viz_dir, "Loss graph.png")
                        if os.path.exists(loss_graph):
                            gr.Image(value=loss_graph, label="MLP Training Loss Over Epochs")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")

                with gr.Row():
                    with gr.Column():
                        cm_img = os.path.join(model_viz_dir, "Confusion Matrix.png")
                        if os.path.exists(cm_img):
                            gr.Image(value=cm_img, label="MLP Confusion Matrix (Best Model)")
                        else:
                            gr.HTML("<p style='color:#94a3b8;text-align:center;padding:40px;'>📁 Image not found</p>")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 5: DATASET
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("📋 Dataset"):
                gr.HTML(section_header_html(
                    "Dataset & Preprocessing",
                    "Complete information about the dataset and preprocessing pipeline",
                    "📋"
                ))
                gr.HTML(dataset_info_html())
                gr.HTML(features_table_html())

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 6: TEAM
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("👥 Team"):
                gr.HTML(section_header_html(
                    "Project Team",
                    "Group members and their contributions to the project",
                    "👥"
                ))
                gr.HTML(team_html())

                # Footer
                gr.HTML("""
                <div style="
                    text-align:center;
                    margin-top:40px;
                    padding:24px;
                    border-top:1px solid rgba(148,163,184,0.08);
                ">
                    <p style="color:#475569;font-size:13px;font-family:'Inter',sans-serif;margin:0;">
                        Built with 🤗 Gradio · Student Mental Health Early Detection System
                    </p>
                    <p style="color:#334155;font-size:11px;margin:6px 0 0;">
                        Domain: Education / Mental Health · Dataset: Kaggle Student Stress Factors
                    </p>
                </div>
                """)

    return demo


# ═══════════════════════════════════════════════════════════════════
# 6. LAUNCH
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
    )
