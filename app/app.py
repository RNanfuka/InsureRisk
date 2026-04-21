"""Streamlit application for InsureRisk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train_model import load_data, preprocess_data  # noqa: E402


st.set_page_config(
    page_title="InsureRisk",
    page_icon="🧠",
    layout="wide",
)


def inject_styles() -> None:
    """Apply custom styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #0f172a;
            --muted: #94a3b8;
            --accent: #14b8a6;
            --navy: #1e293b;
            --teal: #14b8a6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --accent-soft: #ccfbf1;
            --surface: #ffffff;
            --surface-strong: #ffffff;
            --border: rgba(148, 163, 184, 0.22);
            --shadow: 0 18px 44px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.08), transparent 24%),
                radial-gradient(circle at top right, rgba(30, 41, 59, 0.05), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, #f8fafc 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 0.9rem;
            padding-bottom: 1rem;
            max-width: 1380px;
        }

        h1, h2, h3 {
            color: var(--navy);
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.02em;
        }

        .hero {
            background:
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.12), transparent 20%),
                linear-gradient(135deg, #1e293b, #0f172a);
            border-radius: 28px;
            padding: 1.2rem 1.45rem;
            color: white;
            box-shadow: var(--shadow);
            margin-bottom: 0.8rem;
        }

        .hero p {
            font-size: 0.98rem;
            margin: 0.35rem 0 0 0;
            max-width: 44rem;
            color: rgba(241, 245, 249, 0.92);
        }

        .glass-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }

        .metric-card {
            background: var(--surface-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow);
            min-height: 112px;
        }

        .dashboard-kpi {
            background: var(--surface-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 0.85rem 1rem;
            box-shadow: var(--shadow);
        }

        .dashboard-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #475569;
            margin-bottom: 0.3rem;
            font-weight: 700;
        }

        .dashboard-value {
            font-size: 1.6rem;
            font-family: Georgia, "Times New Roman", serif;
            color: var(--navy);
            line-height: 1.05;
        }

        .dashboard-subtext {
            margin-top: 0.25rem;
            color: #475569;
            font-size: 0.88rem;
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #cbd5e1;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }

        .big-number {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.75rem;
            line-height: 1.1;
            color: var(--navy);
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.88rem;
            margin-top: 0.35rem;
        }

        .badge.low {
            background: rgba(34, 197, 94, 0.14);
            color: var(--success);
        }

        .badge.medium {
            background: rgba(245, 158, 11, 0.14);
            color: var(--warning);
        }

        .badge.high {
            background: rgba(239, 68, 68, 0.14);
            color: var(--danger);
        }

        .explain-row {
            border-top: 1px solid rgba(19, 34, 56, 0.08);
            padding: 0.75rem 0 0.2rem 0;
            margin-top: 0.7rem;
        }

        .explain-strong {
            font-weight: 700;
            color: var(--navy);
        }

        .impact-chip {
            display: inline-block;
            margin-top: 0.35rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .impact-chip.up {
            background: rgba(239, 68, 68, 0.12);
            color: var(--danger);
        }

        .impact-chip.down {
            background: rgba(34, 197, 94, 0.12);
            color: var(--success);
        }

        .assumption {
            color: #475569;
            font-size: 0.9rem;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #f1f5f9, #e2e8f0);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }

        div[data-testid="stTabs"] button {
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--navy);
        }

        .stButton > button {
            background: linear-gradient(135deg, #14b8a6, #0f766e);
            color: white;
            border: none;
            box-shadow: 0 10px 24px rgba(20, 184, 166, 0.24);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #0d9488, #115e59);
            color: white;
        }

        div[data-testid="stVerticalBlock"] > div:has(> div .compact-section) {
            gap: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw and processed datasets for defaults and charts."""
    raw_df = load_data(BASE_DIR / "data" / "processed" / "clean_data.csv")
    processed_df = preprocess_data(raw_df)
    return raw_df, processed_df


@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    return joblib.load(BASE_DIR / "models" / "model.pkl")


@st.cache_data
def load_metrics() -> dict:
    """Load saved evaluation metrics if present."""
    metrics_path = BASE_DIR / "reports" / "evaluation_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def dataset_defaults(raw_df: pd.DataFrame) -> dict[str, object]:
    """Compute sensible defaults from the training data."""
    return {
        "age": int(raw_df["age"].median()),
        "bmi": float(raw_df["bmi"].median()),
        "children": int(raw_df["children"].median()),
        "sex": str(raw_df["sex"].mode().iloc[0]),
        "smoker": str(raw_df["smoker"].mode().iloc[0]),
        "region": str(raw_df["region"].mode().iloc[0]),
    }


def build_feature_row(age: int, bmi: float, children: int, smoker: str, sex: str, region: str) -> pd.DataFrame:
    """Convert UI inputs into the model's feature schema."""
    row = pd.DataFrame(
        [
            {
                "age": age,
                "sex": 1 if sex == "female" else 0,
                "bmi": bmi,
                "children": children,
                "smoker": 1 if smoker == "yes" else 0,
                "region_northwest": 1 if region == "northwest" else 0,
                "region_southeast": 1 if region == "southeast" else 0,
                "region_southwest": 1 if region == "southwest" else 0,
            }
        ]
    )
    return row


def align_to_model(frame: pd.DataFrame, model) -> pd.DataFrame:
    """Align columns to the trained model schema."""
    if hasattr(model, "feature_names_in_"):
        return frame.reindex(columns=model.feature_names_in_, fill_value=0)
    return frame


def risk_label(probability: float) -> str:
    """Map probability of the positive class to a risk band."""
    if probability < 0.35:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"


def risk_badge_class(label: str) -> str:
    """Map risk label to CSS class."""
    return label.lower()


def confidence_score(probability: float) -> float:
    """Return confidence in the predicted class."""
    return max(probability, 1 - probability)


def confidence_band(confidence: float) -> str:
    """Map confidence to a plain-English band."""
    if confidence >= 0.90:
        return "High confidence"
    if confidence >= 0.70:
        return "Medium confidence"
    return "Low confidence"


def counterfactual_explanations(
    model,
    current_row: pd.DataFrame,
    baseline_row: pd.DataFrame,
) -> list[dict[str, float | str]]:
    """Estimate feature impact by swapping each input back to a baseline value."""
    current_prob = float(model.predict_proba(current_row)[0][1])
    feature_labels = {
        "age": "Age",
        "bmi": "BMI",
        "children": "Children",
        "smoker": "Smoker status",
    }

    impacts = []
    for feature, label in feature_labels.items():
        test_row = current_row.copy()
        test_row.loc[:, feature] = baseline_row.iloc[0][feature]
        baseline_prob = float(model.predict_proba(test_row)[0][1])
        delta = current_prob - baseline_prob
        impacts.append({"feature": label, "delta": delta})

    impacts.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return impacts


def explanation_sentence(feature: str, delta: float) -> str:
    """Turn a local explanation delta into user-facing text."""
    direction = "raising" if delta >= 0 else "lowering"
    return f"{feature} is {direction} the high-cost risk by {abs(delta) * 100:.1f} percentage points versus baseline."


def impact_chip(delta: float) -> tuple[str, str]:
    """Return chip class and label for an explanation delta."""
    if delta >= 0:
        return "up", "Raises risk"
    return "down", "Lowers risk"


def plain_english_explanation(feature: str, delta: float) -> str:
    """Describe a feature's impact in business-friendly language."""
    magnitude = abs(delta)
    if magnitude >= 0.20:
        strength = "strongly"
    elif magnitude >= 0.08:
        strength = "moderately"
    else:
        strength = "slightly"

    statements = {
        "Age": f"Older age {strength} increases expected insurance cost.",
        "BMI": f"BMI {strength} increases expected insurance cost." if delta >= 0 else f"BMI {strength} lowers expected insurance cost.",
        "Children": f"Having more children has a {strength} impact on expected insurance cost." if magnitude < 0.08 else f"Having more children {strength} increases expected insurance cost." if delta >= 0 else f"Having more children {strength} lowers expected insurance cost.",
        "Smoker status": f"Smoking {strength} increases expected insurance cost." if delta >= 0 else f"Non-smoking status {strength} lowers expected insurance cost.",
    }
    return statements.get(feature, f"{feature} {strength} affects the insurance cost outlook.")


def top_factor_summary(explanation_items: list[dict[str, float | str]]) -> list[str]:
    """Summarize the top factors in ranking form."""
    summaries = []
    for item in explanation_items[:3]:
        delta = abs(float(item["delta"]))
        if delta >= 0.20:
            impact_word = "biggest impact"
        elif delta >= 0.08:
            impact_word = "high impact"
        else:
            impact_word = "moderate impact"
        summaries.append(f"{item['feature']} -> {impact_word}")
    return summaries


def decision_summary(label: str, probability: float) -> str:
    """Create a short business-facing decision summary."""
    if "Lower" in label:
        return f"This customer is likely to have LOWER insurance costs. Estimated probability of high cost: {probability * 100:.1f}%."
    return f"This customer is likely to have HIGHER insurance costs. Estimated probability of high cost: {probability * 100:.1f}%."


def suggested_action(level: str) -> str:
    """Return an underwriting action suggestion."""
    if level == "Low":
        return "Standard pricing"
    if level == "Medium":
        return "Review profile"
    return "Consider premium adjustment"


def feature_importance_table(model) -> pd.DataFrame:
    """Build a global feature importance table when available."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance_df["Importance"] = importance_df["Importance"].round(3)
    return importance_df


def render_feature_chart(importance_df: pd.DataFrame) -> None:
    """Render a themed feature-importance bar chart."""
    chart_df = importance_df.head(6).copy()
    if chart_df.empty:
        st.info("Global feature importance is not available for this model.")
        return

    top_feature = chart_df.iloc[0]["Feature"]
    chart_df["ColorGroup"] = chart_df["Feature"].apply(
        lambda feature: "Top Driver" if feature == top_feature else "Other Drivers"
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Feature:N", sort=None, axis=alt.Axis(title=None, labelAngle=-35, labelColor="#17324d")),
            y=alt.Y("Importance:Q", axis=alt.Axis(title=None, gridColor="#dfe7f0", labelColor="#17324d")),
            color=alt.Color(
                "ColorGroup:N",
                scale=alt.Scale(
                    domain=["Top Driver", "Other Drivers"],
                    range=["#14b8a6", "#1e293b"],
                ),
                legend=None,
            ),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
        )
        .properties(height=250)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    """Render the Streamlit prediction interface."""
    inject_styles()

    model = load_model()
    raw_df, _ = load_reference_data()
    metrics = load_metrics()
    defaults = dataset_defaults(raw_df)

    with st.sidebar:
        st.markdown("## Control Panel")
        st.caption("Set the customer profile, then generate a risk prediction.")
        age = st.slider("Age", min_value=18, max_value=64, value=defaults["age"])
        bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=round(defaults["bmi"], 1), step=0.1)
        smoker = st.selectbox("Smoker", options=["no", "yes"], index=0 if defaults["smoker"] == "no" else 1)
        children = st.slider("Children", min_value=0, max_value=5, value=defaults["children"])

        st.markdown("### Model Inputs")
        sex = st.selectbox("Sex", options=["male", "female"], index=0 if defaults["sex"] == "male" else 1)
        region = st.selectbox(
            "Region",
            options=["northeast", "northwest", "southeast", "southwest"],
            index=["northeast", "northwest", "southeast", "southwest"].index(defaults["region"]),
        )
        predict = st.button("Predict Risk", type="primary", use_container_width=True)
        st.caption("The current model was trained with sex and region, so they remain part of the scoring profile.")

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Phase 5</div>
            <h1 style="margin:0;">InsureRisk Prediction Studio</h1>
            <p>
                Estimate whether a customer profile falls into the high-cost insurance segment,
                surface a simple risk level, and explain the main drivers behind the prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_row = align_to_model(build_feature_row(age, bmi, children, smoker, sex, region), model)
    baseline_row = align_to_model(
        build_feature_row(
            defaults["age"],
            defaults["bmi"],
            defaults["children"],
            defaults["smoker"],
            defaults["sex"],
            defaults["region"],
        ),
        model,
    )

    if predict:
        prediction = int(model.predict(current_row)[0])
        probability = float(model.predict_proba(current_row)[0][1])
        level = risk_label(probability)
        label = "High-Cost Risk" if prediction == 1 else "Lower-Cost Risk"
        explanation_items = counterfactual_explanations(model, current_row, baseline_row)
        confidence = confidence_score(probability)
        confidence_text = confidence_band(confidence)
    else:
        prediction = None
        probability = 0.0
        level = "Waiting"
        label = "Run a profile"
        explanation_items = []
        confidence = 0.0
        confidence_text = "Awaiting prediction"

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4, gap="medium")
    with kpi_1:
        st.markdown(
            f"""
            <div class="dashboard-kpi">
                <div class="dashboard-label">Prediction</div>
                <div class="dashboard-value">{label}</div>
                <div class="dashboard-subtext">Customer cost segment classification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_2:
        display_level = level if predict else "Standby"
        badge_text = "Risk band ready" if predict else "Awaiting score"
        st.markdown(
            f"""
            <div class="dashboard-kpi">
                <div class="dashboard-label">Risk Level</div>
                <div class="dashboard-value">{display_level}</div>
                <div class="dashboard-subtext">{badge_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_3:
        st.markdown(
            f"""
            <div class="dashboard-kpi">
                <div class="dashboard-label">Confidence</div>
                <div class="dashboard-value">{confidence * 100:.1f}%</div>
                <div class="dashboard-subtext">{confidence_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_4:
        st.markdown(
            f"""
            <div class="dashboard-kpi">
                <div class="dashboard-label">Suggested Action</div>
                <div class="dashboard-value">{suggested_action(level) if predict else "Standby"}</div>
                <div class="dashboard-subtext">Recommended next underwriting step</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    content_left, content_right = st.columns([1.05, 0.95], gap="large")

    with content_left:
        st.markdown('<div class="glass-card compact-section">', unsafe_allow_html=True)
        st.subheader("Decision Summary")
        if predict:
            st.write(decision_summary(label, probability))
            st.caption(f"{confidence_text}: the model is {confidence * 100:.1f}% confident in this prediction.")
            st.markdown("**Top factors influencing this decision:**")
            for idx, summary in enumerate(top_factor_summary(explanation_items), start=1):
                st.markdown(f"{idx}. {summary}")
            st.markdown("**Suggested action:**")
            st.write(f"{level} risk -> {suggested_action(level)}")
        else:
            st.info("Use the sidebar to enter a profile, then click `Predict Risk`.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card compact-section" style="margin-top:0.9rem;">', unsafe_allow_html=True)
        st.subheader("Why this prediction")
        if predict:
            st.write(
                f"The model places this profile in the **{label.lower()}** bucket with a **{probability * 100:.1f}%** probability of high insurance cost."
            )
            st.progress(min(max(probability, 0.0), 1.0))
            for item in explanation_items[:3]:
                delta = float(item["delta"])
                chip_class, chip_label = impact_chip(delta)
                st.markdown(
                    f"""
                    <div class="explain-row">
                        <div class="explain-strong">{item["feature"]}</div>
                        <div>{plain_english_explanation(str(item["feature"]), delta)}</div>
                        <span class="impact-chip {chip_class}">{chip_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(f"Inputs used: age {age}, BMI {bmi:.1f}, children {children}, smoker {smoker}, sex {sex}, region {region}.")
        else:
            st.info("Prediction explanations will appear here after scoring a profile.")
        st.markdown("</div>", unsafe_allow_html=True)

    with content_right:
        st.markdown('<div class="glass-card compact-section">', unsafe_allow_html=True)
        st.subheader("What Influenced This Result")
        importance_df = feature_importance_table(model)
        if not importance_df.empty:
            render_feature_chart(importance_df)
        else:
            st.info("Feature influence is not available for this model.")

        if predict:
            st.markdown("**Risk breakdown:**")
            st.write(f"High-cost probability: **{probability * 100:.1f}%**")
            st.write(f"Predicted outcome confidence: **{confidence * 100:.1f}%**")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
