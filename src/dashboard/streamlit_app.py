from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Pharma Safety Lakehouse",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOLD = Path("data/gold")

st.markdown("""
<style>
    .main { background-color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #1e293b; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 0.85rem;
        color: #4ade80;
        margin-top: 2px;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stSelectbox label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "blue":   "#38bdf8",
    "red":    "#f87171",
    "green":  "#4ade80",
    "yellow": "#fbbf24",
    "purple": "#a78bfa",
    "slate":  "#1e293b",
    "border": "#334155",
    "text":   "#f1f5f9",
    "muted":  "#94a3b8",
}

CHART_THEME = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#f1f5f9", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#334155", linecolor="#334155"),
    yaxis=dict(gridcolor="#334155", linecolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
)


@st.cache_data
def load(table: str) -> pd.DataFrame:
    path = GOLD / table
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def kpi(label: str, value, delta: str = ""):
    delta_html = f'<p class="metric-delta">{delta}</p>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## Pharma Safety")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Safety Overview", "Drug Analysis", "Reaction Signals", "ML Monitoring"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<p style="color:#475569;font-size:0.75rem;">openFDA — FAERS Data<br>Bronze / Silver / Gold Pipeline</p>', unsafe_allow_html=True)


if page == "Safety Overview":
    st.markdown("# Safety Overview")
    st.markdown('<p style="color:#94a3b8;margin-top:-12px;">Global adverse-event KPIs and trends</p>', unsafe_allow_html=True)

    overview = load("safety_overview")
    monthly = load("monthly_trends")
    features = load("ml_features")

    if not overview.empty:
        row = overview.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Total Reports", f"{int(row.get('total_reports', 0)):,}")
        with c2:
            kpi("Serious Reports", f"{int(row.get('serious_reports', 0)):,}", delta="")
        with c3:
            kpi("Death Reports", f"{int(row.get('death_reports', 0)):,}")
        with c4:
            rate = row.get("seriousness_rate_pct", 0)
            kpi("Seriousness Rate", f"{rate:.1f}%")

    st.markdown("---")

    if not monthly.empty:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("#### Monthly Report Trend")
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["period"], y=monthly["total_reports"],
                name="Total", line=dict(color=COLORS["blue"], width=2.5),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"
            ))
            fig.add_trace(go.Scatter(
                x=monthly["period"], y=monthly["serious_reports"],
                name="Serious", line=dict(color=COLORS["red"], width=2),
            ))
            fig.update_layout(**CHART_THEME, height=300, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("#### Patient Sex")
            if not features.empty and "patient_sex" in features.columns:
                sex_counts = features["patient_sex"].value_counts().reset_index()
                sex_counts.columns = ["sex", "count"]
                fig2 = px.pie(
                    sex_counts, values="count", names="sex",
                    color_discrete_sequence=[COLORS["blue"], COLORS["purple"], COLORS["muted"]],
                    hole=0.55,
                )
                fig2.update_layout(**CHART_THEME, height=300, showlegend=True)
                fig2.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig2, use_container_width=True)

    if not features.empty and "country" in features.columns:
        st.markdown("#### Top 15 Reporting Countries")
        country_counts = (
            features["country"].value_counts()
            .head(15).reset_index()
        )
        country_counts.columns = ["country", "count"]
        fig3 = px.bar(
            country_counts, x="count", y="country",
            orientation="h",
            color="count",
            color_continuous_scale=[[0, "#1e40af"], [1, "#38bdf8"]],
        )
        fig3.update_layout(**CHART_THEME, height=400, coloraxis_showscale=False)
        fig3.update_yaxes(autorange="reversed", gridcolor="#334155")
        st.plotly_chart(fig3, use_container_width=True)


elif page == "Drug Analysis":
    st.markdown("# Drug Analysis")
    st.markdown('<p style="color:#94a3b8;margin-top:-12px;">Top drugs by report volume and seriousness</p>', unsafe_allow_html=True)

    drugs = load("drug_summary")

    if drugs.empty:
        st.warning("No drug summary data found.")
    else:
        top_n = st.slider("Number of drugs to display", 10, 50, 20)
        top_drugs = drugs.nlargest(top_n, "total_reports")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Drugs by Total Reports")
            fig = px.bar(
                top_drugs.sort_values("total_reports"),
                x="total_reports", y="drug_name",
                orientation="h",
                color="total_reports",
                color_continuous_scale=[[0, "#1e3a5f"], [1, "#38bdf8"]],
            )
            fig.update_layout(**CHART_THEME, height=500, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Seriousness Rate by Drug")
            top_serious = drugs.nlargest(top_n, "seriousness_rate_pct").sort_values("seriousness_rate_pct")
            fig2 = px.bar(
                top_serious,
                x="seriousness_rate_pct", y="drug_name",
                orientation="h",
                color="seriousness_rate_pct",
                color_continuous_scale=[[0, "#7f1d1d"], [1, "#f87171"]],
            )
            fig2.update_layout(**CHART_THEME, height=500, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Drug Detail Table")
        display = drugs.nlargest(top_n, "total_reports")[
            ["drug_name", "total_reports", "serious_reports", "seriousness_rate_pct"]
        ].reset_index(drop=True)
        display.columns = ["Drug", "Total Reports", "Serious", "Seriousness Rate (%)"]
        st.dataframe(display, use_container_width=True, hide_index=True)


elif page == "Reaction Signals":
    st.markdown("# Reaction Signals")
    st.markdown('<p style="color:#94a3b8;margin-top:-12px;">Drug-reaction pairs and potential safety signals</p>', unsafe_allow_html=True)

    pairs = load("drug_reaction_pairs")
    reactions = load("reaction_summary")

    if not reactions.empty:
        st.markdown("#### Top Adverse Reactions")
        top_reactions = reactions.nlargest(20, "total_reports").sort_values("total_reports")
        fig = px.bar(
            top_reactions, x="total_reports", y="reaction_name",
            orientation="h",
            color="seriousness_rate_pct",
            color_continuous_scale=[[0, "#064e3b"], [0.5, "#fbbf24"], [1, "#f87171"]],
            labels={"total_reports": "Total Mentions", "seriousness_rate_pct": "Seriousness %"},
        )
        fig.update_layout(**CHART_THEME, height=450, coloraxis_colorbar=dict(title="Serious %"))
        st.plotly_chart(fig, use_container_width=True)

    if not pairs.empty:
        st.markdown("#### Signal Detection — Drug-Reaction Pairs")
        st.markdown('<p style="color:#94a3b8;font-size:0.85rem;">Top-right quadrant = high frequency AND high seriousness rate = strongest signals</p>', unsafe_allow_html=True)

        min_pairs = st.slider("Minimum pair count", 1, 50, 5)
        filtered = pairs[pairs["pair_count"] >= min_pairs]

        fig2 = px.scatter(
            filtered,
            x="pair_count",
            y="seriousness_rate_pct",
            size="serious_count",
            color="seriousness_rate_pct",
            hover_data=["drug_name", "reaction_name", "pair_count", "serious_count"],
            color_continuous_scale=[[0, "#38bdf8"], [0.5, "#fbbf24"], [1, "#f87171"]],
            labels={
                "pair_count": "Pair Count (frequency)",
                "seriousness_rate_pct": "Seriousness Rate (%)",
            },
        )
        fig2.update_traces(marker=dict(opacity=0.7, line=dict(width=0)))
        fig2.update_layout(**CHART_THEME, height=500,
            coloraxis_colorbar=dict(title="Serious %"),
            shapes=[dict(
                type="rect", xref="paper", yref="paper",
                x0=0.5, y0=0.5, x1=1, y1=1,
                fillcolor="rgba(248,113,113,0.05)",
                line=dict(color="#f87171", width=1, dash="dot"),
            )]
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Top Drug-Reaction Pairs")
        top_pairs = pairs.nlargest(20, "pair_count")[
            ["drug_name", "reaction_name", "pair_count", "serious_count", "seriousness_rate_pct"]
        ].reset_index(drop=True)
        top_pairs.columns = ["Drug", "Reaction", "Pair Count", "Serious", "Seriousness Rate (%)"]
        st.dataframe(top_pairs, use_container_width=True, hide_index=True)


elif page == "ML Monitoring":
    st.markdown("# ML Monitoring")
    st.markdown('<p style="color:#94a3b8;margin-top:-12px;">Seriousness prediction model performance</p>', unsafe_allow_html=True)

    mlflow_runs = load("mlflow_runs")

    if mlflow_runs.empty:
        st.info("No MLflow runs exported yet. Run `python -m src.ml.export_mlflow_metrics` first.")
    else:
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        existing = [c for c in metric_cols if c in mlflow_runs.columns]

        if "roc_auc" in mlflow_runs.columns:
            best = mlflow_runs.loc[mlflow_runs["roc_auc"].idxmax()]
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, metric in zip([c1, c2, c3, c4, c5], existing):
                with col:
                    val = best.get(metric, 0)
                    kpi(metric.upper().replace("_", " "), f"{val:.3f}")

        st.markdown("---")

        if "model_type" in mlflow_runs.columns and existing:
            st.markdown("#### Model Comparison")
            melted = mlflow_runs[["model_type"] + existing].melt(
                id_vars="model_type", var_name="Metric", value_name="Score"
            )
            fig = px.bar(
                melted, x="Metric", y="Score",
                color="model_type", barmode="group",
                color_discrete_sequence=[COLORS["blue"], COLORS["yellow"]],
            )
            fig.update_layout(**CHART_THEME, height=350)
            fig.update_yaxes(range=[0, 1], gridcolor="#334155")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### All Runs")
        display_cols = ["run_name", "model_type", "start_time"] + existing
        display_cols = [c for c in display_cols if c in mlflow_runs.columns]
        st.dataframe(
            mlflow_runs[display_cols].sort_values("roc_auc" if "roc_auc" in mlflow_runs.columns else display_cols[0], ascending=False),
            use_container_width=True,
            hide_index=True,
        )