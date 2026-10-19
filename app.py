from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from model import (
    CONTINUOUS_FEATURES,
    CODE_LABELS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    InputValidationError,
    ModelArtifacts,
    predict_record,
    train_model,
)

st.set_page_config(
    page_title="CardioGuard AI | Heart data, made clear",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAL = "#45e0c1"
BLUE = "#66a9ff"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at 75% -10%, #123548 0, #08111f 42rem); }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -.035em; }
    .block-container { max-width: 1320px; padding-top: 2.4rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #0b1726; border-right: 1px solid #1d3044; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #122438, #0d1928); border: 1px solid #20384a; padding: 1rem 1.2rem; border-radius: 16px; }
    [data-testid="stMetricLabel"] { color: #91a2b9; }
    [data-testid="stMetricValue"] { color: #eff8ff; }
    .eyebrow, .section-kicker { color: #45e0c1; text-transform: uppercase; letter-spacing: .16em; font-size: .75rem; font-weight: 700; }
    .hero { padding: 1.1rem 0 1.5rem; }
    .hero h1 { font-size: clamp(2.7rem, 6vw, 5rem); line-height: 1.02; max-width: 850px; margin: .6rem 0 1rem; }
    .hero p { max-width: 690px; font-size: 1.12rem; color: #a9bad0; line-height: 1.7; }
    .hero-accent { color: #45e0c1; }
    .surface { background: linear-gradient(145deg, #122438, #0e1928); border: 1px solid #22384c; border-radius: 18px; padding: 1.3rem 1.45rem; min-height: 140px; }
    .surface .title { color: #eef8ff; font: 700 1.05rem 'Manrope', sans-serif; margin-bottom: .5rem; }
    .surface p { color: #9fb0c5; line-height: 1.6; margin: 0; }
    .disclaimer { border: 1px solid #78544e; background: #2b211f; border-radius: 12px; padding: 1rem 1.15rem; color: #f5c4b8; line-height: 1.55; }
    .medical-alert { border: 1px solid #873f48; background: #351d28; border-radius: 12px; padding: 1rem 1.15rem; color: #ffd7d2; line-height: 1.55; }
    .stButton>button[kind="primary"] { background: #45e0c1; color: #07131e; border: 0; font-weight: 700; }
    .stButton>button[kind="primary"]:hover { background: #72efd5; color: #07131e; }
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
            min-width: calc(50% - 16px) !important;
            flex-basis: calc(50% - 16px) !important;
        }
    }
    @media (max-width: 420px) {
        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
            min-width: calc(100% - 16px) !important;
            flex-basis: calc(100% - 16px) !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the original Gaussian Naive Bayes model…")
def get_artifacts() -> ModelArtifacts:
    return train_model()


def chart_layout(figure: go.Figure, height: int = 350) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c7d5e5", "family": "DM Sans"},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis={"gridcolor": "#23384b", "zerolinecolor": "#23384b"},
        yaxis={"gridcolor": "#23384b", "zerolinecolor": "#23384b"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    return figure


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='font:800 1.25rem Manrope;color:#eef8ff'>🫀 Cardio<span style='color:#45e0c1'>Guard</span></div>"
            "<div style='color:#91a2b9;font-size:.82rem;margin:.35rem 0 1.5rem'>A machine-learning lab, not a medical device</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Explore",
            ["Overview", "Try the model", "Model analytics", "Dataset explorer", "About & limitations"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Gaussian Naive Bayes · 13 features · local dataset")
        st.markdown(
            "<div class='disclaimer' style='font-size:.82rem'>Educational demonstration only. Never use for diagnosis or care decisions.</div>",
            unsafe_allow_html=True,
        )
    return page


def render_overview(artifacts: ModelArtifacts) -> None:
    metrics = artifacts.metrics
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Responsible ML · transparent by design</div>"
        "<h1>Heart data, made <span class='hero-accent'>clear.</span></h1>"
        "<p>Explore how a classic probabilistic classifier learns from a small included dataset. "
        "Inspect its evaluation, try an educational prediction, and see the limits as clearly as the results.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='disclaimer'><strong>Not medical advice.</strong> This educational demonstration "
        "cannot diagnose heart disease, estimate an individual's real-world risk, or replace a clinician. "
        "If you may be having a heart attack, call your local emergency number immediately.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    columns = st.columns(4)
    columns[0].metric("Dataset records", f"{metrics['total_rows']:,}")
    columns[1].metric("Input features", str(metrics["feature_count"]))
    columns[2].metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
    columns[3].metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.caption("Metrics are recalculated from the notebook's fixed 80/20 split (random_state=42).")
    st.write("")
    cards = st.columns(3)
    card_content = (
        ("01 · Predict", "Enter the same 13 fields the notebook used. The app calls the fitted GaussianNB model, not a mock response."),
        ("02 · Evaluate", "Review holdout metrics, a confusion matrix, and ROC curve from a reproducible split."),
        ("03 · Understand", "Explore dataset-level patterns without exposing individual rows or implying clinical meaning."),
    )
    for column, (title, body) in zip(cards, card_content):
        with column:
            st.markdown(
                f"<div class='surface'><div class='title'>{title}</div><p>{body}</p></div>",
                unsafe_allow_html=True,
            )


def render_prediction(artifacts: ModelArtifacts) -> None:
    st.markdown("<div class='section-kicker'>Interactive prediction</div>", unsafe_allow_html=True)
    st.title("Try the model")
    st.write("Enter the source dataset's feature values. Where the source codebook defines a category, the original code is shown alongside its label.")
    st.markdown(
        "<div class='disclaimer'>This output is a model classification for an educational demonstration, not an assessment of your health. "
        "A displayed probability is the model's native estimate and is not a calibrated personal risk.</div>",
        unsafe_allow_html=True,
    )
    defaults = artifacts.dataset.loc[:, FEATURE_COLUMNS].median()
    values: dict[str, float] = {}
    with st.form("prediction_form"):
        for section, fields in (
            ("Profile", ("age", "sex")),
            ("Clinical measurements", ("trestbps", "chol", "thalach", "oldpeak")),
            ("Encoded fields", ("cp", "fbs", "restecg", "exang", "slope", "ca", "thal")),
        ):
            st.markdown(f"#### {section}")
            field_columns = st.columns(2)
            for index, feature in enumerate(fields):
                with field_columns[index % 2]:
                    minimum, maximum = artifacts.feature_ranges[feature]
                    options = artifacts.feature_values[feature]
                    if feature in CONTINUOUS_FEATURES:
                        values[feature] = st.number_input(
                            FEATURE_LABELS[feature],
                            min_value=minimum,
                            max_value=maximum,
                            value=float(defaults[feature]),
                            step=0.1 if feature == "oldpeak" else 1.0,
                            help=f"Observed dataset range: {minimum:g} to {maximum:g}.",
                        )
                    else:
                        options_list = list(options)
                        default = float(defaults[feature])
                        values[feature] = st.selectbox(
                            FEATURE_LABELS[feature],
                            options=options_list,
                            index=options_list.index(default) if default in options_list else 0,
                            format_func=lambda value, key=feature: (
                                f"{value:g} - {CODE_LABELS[key][int(value)]}"
                                if key in CODE_LABELS and int(value) in CODE_LABELS[key]
                                else f"{value:g} · source code"
                            ),
                        )
        submitted = st.form_submit_button("Run model prediction", type="primary")
    if submitted:
        try:
            prediction, probabilities = predict_record(values, artifacts)
        except InputValidationError as exc:
            st.error(f"Input could not be evaluated: {exc}")
            return
        st.divider()
        st.markdown("### Model output")
        if prediction == 1:
            st.warning(f"The model assigned this record to **dataset target class {prediction}**.")
        else:
            st.success(f"The model assigned this record to **dataset target class {prediction}**.")
        probability_columns = st.columns(2)
        for column, label in zip(probability_columns, (0, 1)):
            if label in probabilities:
                column.metric(f"Model probability · class {label}", f"{probabilities[label]:.1%}")
        st.caption("Target classes are displayed as recorded; no diagnosis or clinical interpretation is implied.")


def render_analytics(artifacts: ModelArtifacts) -> None:
    st.markdown("<div class='section-kicker'>Reproducible evaluation</div>", unsafe_allow_html=True)
    st.title("Model analytics")
    metrics = artifacts.metrics
    st.write(
        f"Gaussian Naive Bayes trained on {metrics['train_rows']} rows and evaluated on "
        f"{metrics['test_rows']} held-out rows. Split: test_size=0.2, random_state=42, "
        "no stratification, matching the notebook."
    )
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    columns[1].metric("Precision · class 1", f"{metrics['precision']:.1%}")
    columns[2].metric("Recall · class 1", f"{metrics['recall']:.1%}")
    columns[3].metric("ROC-AUC · class 1", f"{metrics['roc_auc']:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        matrix = artifacts.confusion
        figure = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[[0, "#122438"], [1, TEAL]],
                showscale=False,
                hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
            )
        )
        figure.update_yaxes(autorange="reversed")
        st.plotly_chart(chart_layout(figure), width="stretch")
    with right:
        st.markdown("#### ROC curve")
        false_positive, true_positive, _ = roc_curve(
            artifacts.y_test, artifacts.y_probability, pos_label=1
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=false_positive, y=true_positive, mode="lines",
                name=f"Model · AUC {metrics['roc_auc']:.3f}",
                line={"color": TEAL, "width": 3},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                line={"color": "#718198", "dash": "dash"},
            )
        )
        figure.update_xaxes(title="False positive rate", range=[0, 1])
        figure.update_yaxes(title="True positive rate", range=[0, 1])
        st.plotly_chart(chart_layout(figure), width="stretch")
    with st.expander("How to interpret these numbers"):
        st.markdown(
            "- **Accuracy:** fraction of held-out records classified correctly.\n"
            "- **Precision (class 1):** fraction of class-1 predictions that were class 1 in this test split.\n"
            "- **Recall (class 1):** fraction of actual class-1 records found by the model.\n"
            "- **ROC-AUC:** ranking performance over thresholds for class 1; it is not clinical utility.\n\n"
            "These estimates come from a small dataset and one fixed split. They are not externally validated "
            "and are not evidence of clinical performance."
        )
    st.caption(
        "Notebook-stored holdout outputs: accuracy 0.8689, precision 0.9000, recall 0.8438. "
        "The metrics above are recomputed from the same split and estimator."
    )


def render_explorer(artifacts: ModelArtifacts) -> None:
    st.markdown("<div class='section-kicker'>Aggregate-only view</div>", unsafe_allow_html=True)
    st.title("Dataset explorer")
    st.write("Explore distributions and feature summaries. Individual records are intentionally not displayed.")
    data = artifacts.dataset
    columns = st.columns(3)
    columns[0].metric("Rows", f"{len(data):,}")
    columns[1].metric("Features", len(FEATURE_COLUMNS))
    columns[2].metric("Missing values", int(data.isna().sum().sum()))
    st.markdown("#### Target-label counts")
    target_counts = (
        data["target"].value_counts().sort_index().rename_axis("Dataset target class")
        .reset_index(name="Records")
    )
    figure = px.bar(
        target_counts, x="Dataset target class", y="Records",
        color="Dataset target class", color_discrete_sequence=[BLUE, TEAL],
    )
    st.plotly_chart(chart_layout(figure, 280), width="stretch")
    selected = st.selectbox("Choose a feature", FEATURE_COLUMNS, format_func=FEATURE_LABELS.get)
    if data[selected].nunique() <= 8:
        frequencies = data[selected].value_counts().sort_index().rename_axis("Source value").reset_index(name="Records")
        figure = px.bar(frequencies, x="Source value", y="Records", color_discrete_sequence=[TEAL])
    else:
        figure = px.histogram(data, x=selected, nbins=24, color_discrete_sequence=[TEAL])
    st.plotly_chart(chart_layout(figure, 320), width="stretch")
    summary = data.loc[:, FEATURE_COLUMNS].describe().T.rename(columns={"50%": "median"})
    summary.insert(0, "Feature", [FEATURE_LABELS[column] for column in summary.index])
    st.markdown("#### Summary statistics")
    st.dataframe(
        summary[["Feature", "count", "mean", "std", "min", "median", "max"]].round(2),
        width="stretch",
        hide_index=True,
    )
    st.caption("Source codes and labels are preserved; aggregate statistics are descriptive only.")


def render_about() -> None:
    st.markdown("<div class='section-kicker'>Transparency & responsible AI</div>", unsafe_allow_html=True)
    st.title("About this project")
    st.write(
        "CardioGuard AI is an educational portfolio app based on this repository's `heart.csv` "
        "and `Naive_Bayed_implementaion.ipynb`. The app reproducibly trains the notebook's "
        "Gaussian Naive Bayes estimator; no serialized model artifact was present."
    )
    st.markdown("#### Model and data facts")
    st.markdown(
        "- **Algorithm:** scikit-learn `GaussianNB`.\n"
        "- **Target:** source column `target`, numeric labels 0 and 1. This app does not assign diagnoses to those labels.\n"
        "- **Features in order:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.\n"
        "- **Preprocessing:** none in the notebook. Values are passed directly to GaussianNB; no imputation, scaling, or recoding is silently introduced.\n"
        "- **Evaluation:** 80/20 `train_test_split`, `random_state=42`, unstratified, as recorded in the notebook."
    )
    st.markdown("#### Limitations")
    st.markdown(
        "- Small dataset; no external or prospective validation.\n"
        "- The historical sample may not represent current populations or any particular person.\n"
        "- GaussianNB assumptions may not fit these features.\n"
        "- Probabilities are model outputs, not calibrated personal risk estimates.\n"
        "- A model score must never replace professional care or delay emergency care."
    )
    st.markdown(
        "<div class='medical-alert'><strong>Possible heart attack symptoms?</strong> Call your local emergency number now. "
        "Do not wait for this application or a model prediction.</div>",
        unsafe_allow_html=True,
    )
    schema = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[column] for column in FEATURE_COLUMNS],
            "CSV column": list(FEATURE_COLUMNS),
            "Input type": [
                "Continuous" if column in CONTINUOUS_FEATURES else "Numeric code"
                for column in FEATURE_COLUMNS
            ],
        }
    )
    st.markdown("#### Feature schema")
    st.dataframe(schema, width="stretch", hide_index=True)


def main() -> None:
    try:
        artifacts = get_artifacts()
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"The application could not load and train the model: {exc}")
        st.info("Check that `heart.csv` is present, readable, and matches the expected schema.")
        st.stop()
    page = sidebar()
    if page == "Overview":
        render_overview(artifacts)
    elif page == "Try the model":
        render_prediction(artifacts)
    elif page == "Model analytics":
        render_analytics(artifacts)
    elif page == "Dataset explorer":
        render_explorer(artifacts)
    else:
        render_about()


if __name__ == "__main__":
    main()
