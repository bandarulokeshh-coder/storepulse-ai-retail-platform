from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import routing as routing_module
from src.classifier import FeedbackAnalyzer, default_analyzer
from src.duplicate_detector import detect_duplicate_groups, duplicate_summary_message
from src.embeddings import embedding_status, ensure_embedding_model, reset_embedding_cache
from src.emerging_detector import build_trend_window, detect_emerging_incidents, summarize_alerts
from src.evaluation import evaluate_emerging_detection, evaluate_feedback_classification, model_insights_summary
from src.policy_engine import evaluate_policy
from src.preprocessing import infer_required_columns, standardize_feedback_frame, compact_text
from src.routing import get_department_for_issue, get_suggested_action, load_taxonomy
from src.storage import initialise_database, load_feedback_frame, upsert_feedback_frame
from src.synthetic_data import CSV_PATH, generate_synthetic_dataset, incident_profiles_dataframe, load_or_create_synthetic_dataset
from src.weekly_report import (
    build_weekly_regional_report,
    export_weekly_report_csv,
    export_weekly_report_pdf,
    priority_formula_description,
)

APP_TITLE = "StorePulse AI"
APP_TAGLINE = "Turning Store Feedback Into Intelligent Action"
BRAND = {
    "primary": "#2563EB",
    "background": "#F8FAFC",
    "sidebar": "#0F172A",
    "success": "#10B981",
    "warning": "#F59E0B",
    "critical": "#EF4444",
    "text": "#0F172A",
    "muted": "#475569",
}

PAGES = [
    "Dashboard",
    "AI Feedback Analyzer",
    "Emerging Alerts",
    "Weekly Regional Reports",
    "Store Analytics",
    "Evaluation & Model Insights",
    "Settings / About",
]

SCENARIOS: dict[str, dict[str, str]] = {
    "Multilingual Feedback": {
        "text": "Store lo freezer work avvatledu, milk packets spoil avtunnayi!",
        "page": "AI Feedback Analyzer",
        "expected": "Identify refrigeration failure and potential spoilage.",
    },
    "Safety Escalation": {
        "text": "Wrong allergen label on food packet.",
        "page": "AI Feedback Analyzer",
        "expected": "HIGH severity and a responsible department via the policy engine.",
    },
    "Emerging Incident": {
        "text": "",
        "page": "Emerging Alerts",
        "expected": "Seeded store-level incident rising across weeks, with supporting feedback.",
    },
    "Regional Report": {
        "text": "",
        "page": "Weekly Regional Reports",
        "expected": "Downloadable weekly regional action brief (PDF/CSV).",
    },
}

DEFAULT_GROUND_TRUTH = {"issue_type": True, "severity": True, "actionability": True}


st.set_page_config(page_title=APP_TITLE, page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
      :root {{
        --primary: {BRAND['primary']};
        --background: {BRAND['background']};
        --sidebar: {BRAND['sidebar']};
        --success: {BRAND['success']};
        --warning: {BRAND['warning']};
        --critical: {BRAND['critical']};
        --text: {BRAND['text']};
        --muted: {BRAND['muted']};
      }}
      .stApp {{ background: var(--background); color: var(--text); }}
      section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0F172A 0%, #111827 100%); }}
      section[data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}
      .hero {{
        padding: 1.4rem 1.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(37,99,235,0.96), rgba(15,23,42,0.94));
        color: white;
        box-shadow: 0 24px 60px rgba(15,23,42,0.18);
      }}
      .hero h1 {{ margin: 0; font-size: 2.1rem; line-height: 1.05; }}
      .hero p {{ margin: 0.5rem 0 0; opacity: 0.94; }}
      .metric-card {{
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem;
        background: #fff;
        border: 1px solid #E2E8F0;
        box-shadow: 0 8px 28px rgba(15,23,42,0.08);
      }}
      .metric-label {{ color: #64748B; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .08em; }}
      .metric-value {{ font-size: 2rem; font-weight: 800; color: #0F172A; margin-top: 0.15rem; }}
      .metric-note {{ color: #475569; font-size: 0.82rem; margin-top: 0.25rem; }}
      .pill {{ display:inline-block; padding:0.25rem 0.65rem; border-radius:999px; font-size:0.78rem; font-weight:700; }}
      .pill-high {{ background: rgba(239,68,68,0.12); color: #B91C1C; }}
      .pill-medium {{ background: rgba(245,158,11,0.14); color: #B45309; }}
      .pill-low {{ background: rgba(16,185,129,0.12); color: #047857; }}
      .soft-card {{ border-radius: 20px; background: white; border: 1px solid #E2E8F0; padding: 1rem; box-shadow: 0 8px 24px rgba(15,23,42,0.05); }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_taxonomy() -> dict[str, Any]:
    return load_taxonomy()


@st.cache_data(show_spinner=False)
def load_demo_bundle() -> pd.DataFrame:
    bundle = load_or_create_synthetic_dataset()
    df = bundle.dataframe.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "week_start" not in df.columns and "timestamp" in df.columns:
        df["week_start"] = df["timestamp"].dt.to_period("W-MON").dt.start_time
    initialise_database(df)
    return df


@st.cache_resource(show_spinner=False)
def get_analyzer(df_signature: str) -> FeedbackAnalyzer:
    analyzer = default_analyzer()
    bundle_df = st.session_state.get("active_dataframe")
    if isinstance(bundle_df, pd.DataFrame) and not bundle_df.empty:
        analyzer.fit_from_dataframe(bundle_df)
    labeled_df = st.session_state.get("labeled_training_dataframe")
    if isinstance(labeled_df, pd.DataFrame) and not labeled_df.empty:
        # Official labeled data wins over demo labels for every target it provides.
        analyzer.fit_from_dataframe(labeled_df)
    analyzer.set_use_embeddings(bool(st.session_state.get("use_embeddings", True)))
    return analyzer


@st.cache_data(show_spinner=False)
def compute_emerging_alerts(df: pd.DataFrame, minimum_support: int, growth_threshold: float, spike_threshold: float) -> pd.DataFrame:
    return summarize_alerts(
        detect_emerging_incidents(
            df,
            minimum_support=minimum_support,
            growth_threshold=growth_threshold,
            spike_threshold=spike_threshold,
        )
    )


@st.cache_data(show_spinner=False)
def compute_duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    return detect_duplicate_groups(df)


@st.cache_data(show_spinner=False)
def compute_evaluation_bundle(
    df: pd.DataFrame,
    minimum_support: int,
    growth_threshold: float,
    spike_threshold: float,
    ground_truth: dict[str, bool] | None = None,
    analyzer_signature: str = "",
) -> dict:
    # The analyzer has already been fitted by get_analyzer() for this signature;
    # do not refit here so uploaded labeled data is not overwritten.
    analyzer = default_analyzer()
    gt = {**DEFAULT_GROUND_TRUTH, **(ground_truth or {})}
    alerts = detect_emerging_incidents(
        df,
        minimum_support=minimum_support,
        growth_threshold=growth_threshold,
        spike_threshold=spike_threshold,
    )
    return {
        "classification": evaluate_feedback_classification(df, analyzer, ground_truth=gt),
        "emerging": evaluate_emerging_detection(
            df,
            alerts,
            actionability_labels_available=bool(gt.get("actionability", False)),
        ),
        "model_insights": model_insights_summary(analyzer),
    }



def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])



def prepare_active_dataframe() -> pd.DataFrame:
    if "active_dataframe" not in st.session_state:
        demo = load_demo_bundle()
        st.session_state["active_dataframe"] = demo
        st.session_state["active_source"] = "Synthetic Demonstration Data"
        set_ground_truth(issue_type=True, severity=True, actionability=True)
    return st.session_state["active_dataframe"].copy()



def set_active_dataframe(df: pd.DataFrame, source: str) -> None:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.loc[~data["timestamp"].isna()].copy()
    if "week_start" not in data.columns:
        data["week_start"] = data["timestamp"].dt.to_period("W-MON").dt.start_time
    st.session_state["active_dataframe"] = data
    st.session_state["active_source"] = source
    initialise_database(data)



def filter_dataframe(
    df: pd.DataFrame,
    region: str | None,
    store_id: str | None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
    channel: str | None,
    severity: str | None,
    issue_category: str | None,
) -> pd.DataFrame:
    frame = df.copy()
    if region and region != "All":
        frame = frame[frame["region"] == region]
    if store_id and store_id != "All":
        frame = frame[frame["store_id"] == store_id]
    if channel and channel != "All":
        frame = frame[frame["channel"] == channel]
    if severity and severity != "All":
        frame = frame[frame["severity"].astype(str).str.upper() == severity.upper()]
    if issue_category and issue_category != "All":
        frame = frame[frame["issue_category"] == issue_category]
    if date_range is not None and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    return frame.sort_values("timestamp")



def metric_card(label: str, value: Any, note: str, accent: str = BRAND["primary"]) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{accent};">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def display_metrics(kpis: list[tuple[str, Any, str, str]]) -> None:
    cols = st.columns(len(kpis))
    for col, (label, value, note, accent) in zip(cols, kpis):
        with col:
            metric_card(label, value, note, accent)


def get_ground_truth() -> dict[str, bool]:
    """Which ground-truth label columns exist for the active dataset."""
    return {**DEFAULT_GROUND_TRUTH, **st.session_state.get("ground_truth", {})}


def set_ground_truth(issue_type: bool, severity: bool, actionability: bool) -> None:
    st.session_state["ground_truth"] = {
        "issue_type": bool(issue_type),
        "severity": bool(severity),
        "actionability": bool(actionability),
    }


def show_embedding_pipeline_status(load_if_needed: bool = True) -> None:
    """Display which classification pipeline is active (embeddings vs rule fallback)."""
    if not st.session_state.get("use_embeddings", True):
        st.caption("Pipeline: rule-based multilingual baseline (embeddings disabled in Settings).")
        return
    status = embedding_status()
    if not status.available and not status.attempted and load_if_needed:
        with st.spinner("Loading multilingual embedding model (first run may download it)..."):
            ensure_embedding_model()
        status = embedding_status()
    if status.available:
        st.caption(f"Pipeline: rule-based baseline + multilingual embeddings ({status.model_name}).")
    else:
        st.warning(
            f"Multilingual embedding model unavailable — {status.reason}. "
            "Falling back to the labeled rule-based baseline."
        )


def clear_taxonomy_caches() -> None:
    """Clear every cache that may hold the previous issue taxonomy."""
    routing_module.load_taxonomy.cache_clear()
    routing_module.get_department_for_issue.cache_clear()
    routing_module.get_suggested_action.cache_clear()
    routing_module.get_default_severity.cache_clear()
    routing_module.category_keywords.cache_clear()
    default_analyzer().reload_taxonomy()
    if hasattr(get_taxonomy, "clear"):
        get_taxonomy.clear()
    if hasattr(get_analyzer, "clear"):
        get_analyzer.clear()



def make_weekly_trend_chart(df: pd.DataFrame, title: str = "Feedback Trends") -> go.Figure:
    trend = df.copy()
    trend["week_start"] = trend["timestamp"].dt.to_period("W-MON").dt.start_time
    trend = trend.groupby("week_start").size().reset_index(name="feedback_count").sort_values("week_start")
    fig = px.line(trend, x="week_start", y="feedback_count", markers=True, title=title)
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Week", yaxis_title="Feedback Count")
    return fig



def make_issue_distribution_chart(df: pd.DataFrame) -> go.Figure:
    issue_counts = df[df["issue_category"].isin([c for c in df["issue_category"].unique() if c not in ["Customer Praise", "General Feedback"]])]
    issue_counts = issue_counts.groupby("issue_category").size().reset_index(name="count").sort_values("count", ascending=False).head(12)
    fig = px.bar(issue_counts, x="count", y="issue_category", orientation="h", title="Issue Distribution")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Complaints", yaxis_title="Issue Category")
    return fig



def make_department_workload_chart(df: pd.DataFrame) -> go.Figure:
    departments = ["Store Operations", "Supply Chain", "Merchandising", "Digital", "HR"]
    data = df[df["department"].isin(departments) & (df["actionable"].astype(bool))].groupby("department").size().reindex(departments, fill_value=0).reset_index(name="actionable_issues")
    fig = px.bar(data, x="department", y="actionable_issues", color="department", title="Department Workload")
    fig.update_layout(template="plotly_white", height=320, showlegend=False, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Department", yaxis_title="Actionable Issues")
    return fig



def make_issue_top5_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Issue Category", "Severity", "Complaints", "Trend", "Affected Stores", "Department"])
    frame = df.copy()
    frame["week_start"] = frame["timestamp"].dt.to_period("W-MON").dt.start_time
    current_week = frame["week_start"].max()
    previous_week = current_week - pd.Timedelta(weeks=1)
    current = frame[frame["week_start"] == current_week]
    prev = frame[frame["week_start"] == previous_week]
    current_counts = current.groupby("issue_category").size()
    prev_counts = prev.groupby("issue_category").size()
    rows = []
    for issue, count in current_counts.sort_values(ascending=False).head(5).items():
        past = int(prev_counts.get(issue, 0))
        trend = (count + 1) / (past + 1)
        subset = current[current["issue_category"] == issue]
        rows.append(
            {
                "Issue Category": issue,
                "Severity": subset["severity"].mode().iloc[0],
                "Complaints": int(count),
                "Trend": f"{'↑' if trend > 1.15 else '↓' if trend < 0.9 else '→'} {trend:.2f}x",
                "Affected Stores": int(subset["store_id"].nunique()),
                "Department": subset["department"].mode().iloc[0],
            }
        )
    return pd.DataFrame(rows)



def render_dashboard(df: pd.DataFrame, analyzer: FeedbackAnalyzer, minimum_support: int, growth_threshold: float, spike_threshold: float) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{APP_TITLE}</h1>
          <p>{APP_TAGLINE} — multilingual feedback intelligence, safety policy escalation, and early-warning detection for retail teams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered = df.copy()
    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
        regions = ["All"] + sorted(filtered["region"].dropna().astype(str).unique().tolist())
        stores = ["All"] + sorted(filtered["store_id"].dropna().astype(str).unique().tolist())
        channels = ["All"] + sorted(filtered["channel"].dropna().astype(str).unique().tolist())
        severities = ["All", "LOW", "MEDIUM", "HIGH"]
        categories = ["All"] + sorted(filtered["issue_category"].dropna().astype(str).unique().tolist())
        min_date = filtered["timestamp"].min().date() if not filtered.empty else pd.Timestamp.today().date()
        max_date = filtered["timestamp"].max().date() if not filtered.empty else pd.Timestamp.today().date()
        with col1:
            region = st.selectbox("Region", regions, index=0)
        with col2:
            store_id = st.selectbox("Store", stores, index=0)
        with col3:
            channel = st.selectbox("Channel", channels, index=0)
        with col4:
            severity = st.selectbox("Severity", severities, index=0)
        with col5:
            issue_category = st.selectbox("Issue Category", categories, index=0)
        with col6:
            date_range = st.date_input("Date Range", value=(min_date, max_date))

    filtered = filter_dataframe(filtered, region, store_id, tuple(date_range) if isinstance(date_range, (list, tuple)) else None, channel, severity, issue_category)
    alerts = compute_emerging_alerts(filtered, minimum_support, growth_threshold, spike_threshold)
    duplicate_groups = compute_duplicate_groups(filtered)

    actionable_pct = round((filtered[filtered["actionable"].astype(bool)].shape[0] / max(len(filtered), 1)) * 100, 1)
    department_attention = int(filtered[filtered["actionable"].astype(bool)]["department"].nunique())
    display_metrics(
        [
            ("Total Feedback Analyzed", f"{len(filtered):,}", "Records after current filters", BRAND["primary"]),
            ("High-Severity Issues", f"{int((filtered['severity'].astype(str).str.upper() == 'HIGH').sum()):,}", "Policy-first escalations", BRAND["critical"]),
            ("Emerging Incidents", f"{len(alerts):,}", "Store-level spike alerts", BRAND["warning"]),
            ("Stores Monitored", f"{filtered['store_id'].nunique():,}", "Unique stores in view", BRAND["primary"]),
            ("Actionable Feedback %", f"{actionable_pct:.1f}%", "Feedback flagged for action", BRAND["success"]),
            ("Departments Requiring Attention", f"{department_attention:,}", "Departments with actionable items", BRAND["critical"]),
        ]
    )

    st.markdown("### Weekly Top 5 Operational Issues")
    top5 = make_issue_top5_table(filtered)
    if top5.empty:
        st.info("No issue data available for the selected filters.")
    else:
        st.dataframe(top5, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_weekly_trend_chart(filtered), width="stretch")
    with c2:
        st.plotly_chart(make_issue_distribution_chart(filtered), width="stretch")
    st.plotly_chart(make_department_workload_chart(filtered), width="stretch")

    st.markdown("### Duplicate Screening")
    if duplicate_groups.empty:
        st.success("No probable duplicate incident groups were found in the current filtered view.")
    else:
        st.warning(duplicate_summary_message(duplicate_groups))
        st.dataframe(
            duplicate_groups[["group_id", "store_id", "issue_category", "record_count", "channels", "first_seen", "last_seen"]],
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Demo Mode")
    st.caption("Four judge-ready scenarios — every one runs through live application logic.")
    demo_cols = st.columns(4)
    for idx, (name, scenario) in enumerate(SCENARIOS.items()):
        with demo_cols[idx]:
            if st.button(name, key=f"demo_{idx}", help=scenario["expected"]):
                if scenario["text"]:
                    st.session_state["analyzer_text"] = scenario["text"]
                st.session_state["selected_page"] = scenario["page"]
                st.rerun()



def render_analyzer(analyzer: FeedbackAnalyzer) -> None:
    st.subheader("AI Feedback Analyzer")
    st.write("Paste multilingual customer or store feedback and let StorePulse AI classify the issue, severity, and routing.")
    show_embedding_pipeline_status()

    examples = [scenario["text"] for scenario in SCENARIOS.values() if scenario["text"]]
    if "analyzer_text" not in st.session_state:
        st.session_state["analyzer_text"] = examples[0]

    sample_cols = st.columns(len(examples))
    for idx, sample in enumerate(examples):
        with sample_cols[idx]:
            if st.button(f"Load Example {idx + 1}", key=f"sample_{idx}"):
                st.session_state["analyzer_text"] = sample

    text = st.text_area("Feedback text", value=st.session_state["analyzer_text"], height=140)
    st.session_state["analyzer_text"] = text

    if st.button("Analyze Feedback", type="primary") or text:
        result = analyzer.analyze(text)
        policy = evaluate_policy(text, result.issue_category)
        cols = st.columns(4)
        cols[0].metric("Detected Issue", result.issue_category)
        cols[1].metric("Severity", result.severity)
        cols[2].metric("Actionable", "YES" if result.actionable else "NO")
        cols[3].metric("Responsible Department", result.department)

        st.markdown(f"**Suggested Action:** {result.suggested_action}")
        st.markdown(f"**Explanation:** {result.explanation}")
        st.markdown(f"**Classification Method:** `{result.method}`")
        if result.confidence is not None:
            st.caption(f"Model confidence: {result.confidence:.2f}" if result.method != "rule_based" else f"Rule-match strength: {result.confidence:.2f}")
        else:
            st.caption("No confidence score shown for rule-based or neutral feedback decisions.")
        if policy.human_review_required:
            st.warning("Ambiguous safety language detected — human verification recommended.")

        st.markdown("#### Why this result?")
        if result.rule_matches:
            st.write(f"Rule matches: {', '.join(result.rule_matches)}")
        else:
            st.write("No strong issue-rule match was found; the analyzer used fallback logic.")
        st.write(f"Policy reason: {policy.reason}")



def render_alerts(df: pd.DataFrame, minimum_support: int, growth_threshold: float, spike_threshold: float) -> pd.DataFrame:
    st.subheader("Emerging Alerts")
    st.write("Early-warning system based on store-level normalized complaint rates, robust baselines, and seeded incident evaluation.")
    alerts = compute_emerging_alerts(df, minimum_support, growth_threshold, spike_threshold)
    if alerts.empty:
        st.info("No alerts were triggered for the current dataset and thresholds.")
        return alerts

    alerts = alerts.sort_values(["alert_date", "growth_indicator", "current_count"], ascending=[False, False, False])
    st.success(f"{len(alerts)} emerging alerts detected.")
    st.dataframe(
        alerts[["store_id", "region", "issue_category", "severity", "department", "current_count", "normalized_rate", "previous_baseline", "growth_indicator", "alert_date"]],
        width="stretch",
        hide_index=True,
    )

    selected = st.selectbox(
        "Inspect alert",
        alerts.index.tolist(),
        format_func=lambda idx: f"{alerts.loc[idx, 'store_id']} | {alerts.loc[idx, 'issue_category']} | {pd.to_datetime(alerts.loc[idx, 'alert_date']).date()}",
    )
    alert_row = alerts.loc[selected]
    trend = build_trend_window(df, alert_row["store_id"], alert_row["issue_category"], weeks=8)
    if not trend.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["week_start"], y=trend["issue_count"], mode="lines+markers", name="Issue Count"))
        fig.add_trace(go.Scatter(x=trend["week_start"], y=trend["normalized_rate"], mode="lines+markers", name="Normalized Rate", yaxis="y2"))
        fig.update_layout(
            template="plotly_white",
            title=f"{alert_row['store_id']} — {alert_row['issue_category']} Trend",
            yaxis=dict(title="Complaint Count"),
            yaxis2=dict(title="Normalized Rate", overlaying="y", side="right"),
            height=420,
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Alert Explanation")
    st.write(alert_row["explanation"])
    st.markdown("#### Supporting Feedback Quotes")
    quotes = alert_row.get("supporting_quotes", [])
    if isinstance(quotes, list) and quotes:
        for quote in quotes:
            st.write(f"• {quote}")
    else:
        st.write("No supporting quotes available for this alert.")
    return alerts



def render_reports(df: pd.DataFrame, alerts: pd.DataFrame) -> None:
    st.subheader("Weekly Regional Reports")
    regions = sorted(df["region"].dropna().astype(str).unique().tolist())
    if not regions:
        st.warning("No regions available in the current dataset.")
        return
    region = st.selectbox("Region", regions)
    week_options = sorted(df[df["region"] == region]["week_start"].dropna().unique())
    if not week_options:
        st.warning("No weekly data available for the selected region.")
        return
    selected_week = st.selectbox("Reporting week", week_options, format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
    report = build_weekly_regional_report(df, alerts, region=region, week_start=pd.Timestamp(selected_week))

    cols = st.columns(4)
    cols[0].metric("Total Feedback", report.total_feedback)
    cols[1].metric("High-Severity Issues", report.high_severity_issues)
    cols[2].metric("Emerging Alerts", report.emerging_alerts)
    cols[3].metric("Top Issue", report.overview.get("top_issue", "—"))

    st.caption(report.formula)
    if report.top_issues.empty:
        st.info("No priority issues detected for this region and week.")
    else:
        st.dataframe(report.top_issues, width="stretch", hide_index=True)

    pdf_path = Path("data") / f"storepulse_report_{region}_{pd.Timestamp(selected_week).strftime('%Y%m%d')}.pdf"
    csv_path = Path("data") / f"storepulse_report_{region}_{pd.Timestamp(selected_week).strftime('%Y%m%d')}.csv"
    pdf_buffer = BytesIO()
    csv_buffer = BytesIO()
    export_weekly_report_pdf(report, pdf_path)
    export_weekly_report_csv(report, csv_path)
    st.download_button(
        "Download PDF",
        data=pdf_path.read_bytes(),
        file_name=pdf_path.name,
        mime="application/pdf",
    )
    st.download_button(
        "Download CSV",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
    )



def render_store_analytics(df: pd.DataFrame) -> None:
    st.subheader("Store Analytics")
    stores = sorted(df["store_id"].dropna().astype(str).unique().tolist())
    if not stores:
        st.warning("No stores available in the current dataset.")
        return
    store_id = st.selectbox("Store", stores)
    subset = df[df["store_id"] == store_id].copy()
    if subset.empty:
        st.info("No data found for the selected store.")
        return

    cols = st.columns(4)
    cols[0].metric("Feedback Records", len(subset))
    cols[1].metric("Regions", subset["region"].nunique())
    cols[2].metric("Actionable %", f"{subset['actionable'].astype(bool).mean() * 100:.1f}%")
    cols[3].metric("High Severity", int((subset['severity'].astype(str).str.upper() == 'HIGH').sum()))

    trend = subset.groupby(subset["timestamp"].dt.to_period("W-MON").dt.start_time).size().reset_index(name="feedback_count")
    fig = px.line(trend, x="timestamp", y="feedback_count", markers=True, title=f"Store {store_id} Feedback Trend")
    fig.update_layout(template="plotly_white", height=320)
    st.plotly_chart(fig, width="stretch")

    issue_breakdown = subset.groupby("issue_category").size().reset_index(name="count").sort_values("count", ascending=False).head(10)
    st.plotly_chart(px.bar(issue_breakdown, x="count", y="issue_category", orientation="h", title="Issue Breakdown"), width="stretch")

    duplicate_groups = detect_duplicate_groups(subset)
    st.markdown("#### Potential Duplicate Groups")
    if duplicate_groups.empty:
        st.success("No probable duplicate groups found for this store.")
    else:
        st.warning(f"{len(duplicate_groups)} potential duplicate groups detected for this store.")
        st.dataframe(duplicate_groups, width="stretch", hide_index=True)

    st.markdown("#### Recent Feedback")
    st.dataframe(
        subset[["timestamp", "channel", "issue_category", "severity", "actionable", "feedback_text"]].sort_values("timestamp", ascending=False).head(20),
        width="stretch",
        hide_index=True,
    )



def render_evaluation(df: pd.DataFrame, minimum_support: int, growth_threshold: float, spike_threshold: float) -> None:
    st.subheader("Evaluation & Model Insights")
    st.write(
        "Metrics are demonstration-only when operating on synthetic records. "
        "Targets without ground-truth labels are reported as unavailable instead of being estimated."
    )
    ground_truth = get_ground_truth()
    with st.spinner("Computing evaluation metrics..."):
        bundle = compute_evaluation_bundle(
            df,
            minimum_support,
            growth_threshold,
            spike_threshold,
            ground_truth=ground_truth,
            analyzer_signature=st.session_state.get("analyzer_signature", ""),
        )
    classification = bundle["classification"]
    emerging = bundle["emerging"]
    insights = bundle["model_insights"]

    def _f1(entry: dict[str, Any]) -> str:
        return f"{entry['metrics'].macro_f1:.3f}" if entry["available"] else "—"

    def _acc_note(entry: dict[str, Any]) -> str:
        if entry["available"]:
            return f"Accuracy {entry['metrics'].accuracy:.3f}"
        return "Labels required"

    kpis = [
        ("Issue Type Macro F1", _f1(classification["issue_type"]), _acc_note(classification["issue_type"]), BRAND['primary']),
        ("Severity Macro F1", _f1(classification["severity"]), _acc_note(classification["severity"]), BRAND['warning']),
        ("Actionability Macro F1", _f1(classification["actionability"]), _acc_note(classification["actionability"]), BRAND['success']),
        ("True Incidents Detected", f"{emerging['true_incidents_detected']}/{emerging['total_known_incidents']}", "Seeded incident coverage", BRAND['critical']),
        ("Detection Lead Time", f"{emerging['median_detection_lead_time_days'] if emerging['median_detection_lead_time_days'] is not None else '—'} days", "Median lead-time", BRAND['primary']),
        ("Escalation Precision", f"{emerging['escalation_precision']:.3f}" if emerging['escalation_precision'] is not None else "—", "Only when labels exist", BRAND['critical']),
    ]
    display_metrics(kpis)

    st.markdown("### Confusion Matrices")
    tabs = st.tabs(["Issue Type", "Severity", "Actionability"])
    for tab, key in zip(tabs, ["issue_type", "severity", "actionability"]):
        with tab:
            entry = classification[key]
            if entry["available"]:
                if entry["demo_only"]:
                    st.caption("Demonstration-only metrics computed on synthetic labels.")
                st.dataframe(entry["metrics"].confusion, width="stretch")
            else:
                st.info(entry["message"])

    st.markdown("### Model Insights")
    st.json(insights)
    precision_note = emerging.get("escalation_precision_note")
    if precision_note:
        st.info(precision_note)
    else:
        st.info("Escalation precision = true actionable escalations / total escalations. For synthetic data this value is demonstration-only.")



def render_settings(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Settings / About")
    st.write("StorePulse AI is built for the TalentAP Mega Hackathon 2026 retail problem statement R5 — Turning Store Chatter Into Action.")
    st.markdown("**Technology stack:** Streamlit, Pandas, NumPy, Scikit-learn, Plotly, SQLite, ReportLab, and optional sentence-transformers.")
    st.markdown("**Early-warning thresholds:** tune the alert sensitivity from the sidebar.")
    st.code(priority_formula_description())

    st.markdown("#### Multilingual Embeddings")
    current_embeddings = st.session_state.get("use_embeddings", True)
    use_embeddings = st.toggle(
        "Use multilingual sentence embeddings",
        value=current_embeddings,
        help="Uses a local sentence-transformers multilingual model when available; falls back to the labeled rule-based baseline if it cannot load.",
    )
    if use_embeddings != current_embeddings:
        st.session_state["use_embeddings"] = use_embeddings
        if use_embeddings:
            with st.spinner("Loading multilingual embedding model..."):
                ensure_embedding_model()
    emb_status = embedding_status()
    if emb_status.available:
        st.success(f"Embeddings active: {emb_status.model_name}")
    elif emb_status.attempted:
        st.warning(f"Embeddings unavailable — {emb_status.reason}. The rule-based fallback remains active.")
    else:
        st.info("Embedding model not loaded yet; it loads on first use (or use the retry button).")
    if st.button("Retry embedding model load"):
        reset_embedding_cache()
        with st.spinner("Retrying model load..."):
            ensure_embedding_model()
        st.rerun()

    st.markdown("#### Taxonomy Preview")
    taxonomy = get_taxonomy()
    st.write(f"{len(taxonomy.get('categories', []))} operational categories loaded from `config/issue_taxonomy.json`.")
    preview = pd.DataFrame(taxonomy.get("categories", []))[["issue_category", "department", "severity_default"]].head(12)
    st.dataframe(preview, width="stretch", hide_index=True)

    st.markdown("#### Optional Uploads (Official Dataset Mode)")
    st.caption("Additional official hackathon files. These never silently replace synthetic demo data.")

    labeled_file = st.file_uploader(
        "Labeled training data (CSV with feedback_text + issue_category / severity / actionable)",
        type=["csv"],
        key="labeled_uploader",
    )
    if labeled_file is not None and st.button("Train on Labeled Data"):
        labeled_df = pd.read_csv(labeled_file)
        if "feedback_text" not in labeled_df.columns:
            st.error("Labeled file must contain a feedback_text column.")
        else:
            analyzer = default_analyzer()
            analyzer.fit_from_dataframe(labeled_df)
            if analyzer.model_bundle.trained_targets:
                st.session_state["labeled_training_dataframe"] = labeled_df
                st.success(f"Trained supervised models for: {', '.join(analyzer.model_bundle.trained_targets)}.")
            else:
                st.warning("No usable labels found (each target needs at least two classes). Rule-based fallback remains active.")

    master_file = st.file_uploader(
        "Store master (CSV with store_id, and optional region / store_size)",
        type=["csv"],
        key="store_master_uploader",
    )
    if master_file is not None and st.button("Apply Store Master"):
        master_df = pd.read_csv(master_file)
        if "store_id" not in master_df.columns:
            st.error("Store master must contain a store_id column.")
        else:
            active = st.session_state.get("active_dataframe")
            if isinstance(active, pd.DataFrame) and not active.empty:
                updated = active.copy()
                mapping_frame = master_df.drop_duplicates("store_id").set_index("store_id")
                for column in ["region", "store_size"]:
                    if column in mapping_frame.columns:
                        mapped = updated["store_id"].map(mapping_frame[column])
                        updated[column] = mapped.fillna(updated[column])
                set_active_dataframe(updated, st.session_state.get("active_source", "Uploaded Official Dataset"))
                st.success("Store master applied to the active dataset.")
                st.rerun()

    taxonomy_file = st.file_uploader(
        "Issue taxonomy (JSON with the same schema as config/issue_taxonomy.json)",
        type=["json"],
        key="taxonomy_uploader",
    )
    if taxonomy_file is not None and st.button("Replace Issue Taxonomy"):
        try:
            payload = json.loads(taxonomy_file.getvalue().decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("categories"), list):
                st.error("Taxonomy JSON must be an object with a 'categories' list.")
            else:
                config_path = Path("config") / "issue_taxonomy.json"
                backup_path = Path("config") / "issue_taxonomy.backup.json"
                if config_path.exists() and not backup_path.exists():
                    backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
                config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                clear_taxonomy_caches()
                st.success("Taxonomy updated; classification and routing caches cleared.")
                st.rerun()
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
    backup_path = Path("config") / "issue_taxonomy.backup.json"
    if backup_path.exists() and st.button("Restore previous taxonomy"):
        (Path("config") / "issue_taxonomy.json").write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
        clear_taxonomy_caches()
        st.success("Previous taxonomy restored.")
        st.rerun()

    st.markdown("#### Upload Official Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        required = infer_required_columns(raw_df.columns)
        st.write("Detected column mapping:", required)
        text_col = st.selectbox("Feedback text column", raw_df.columns, index=list(raw_df.columns).index(required["text"]) if required["text"] in raw_df.columns else 0)
        date_col = st.selectbox("Date column", raw_df.columns, index=list(raw_df.columns).index(required["date"]) if required["date"] in raw_df.columns else 0)
        store_col = st.selectbox("Store ID column", raw_df.columns, index=list(raw_df.columns).index(required["store_id"]) if required["store_id"] in raw_df.columns else 0)
        region_col = st.selectbox("Region column", raw_df.columns, index=list(raw_df.columns).index(required["region"]) if required["region"] in raw_df.columns else 0)
        channel_col = st.selectbox("Channel column", raw_df.columns, index=list(raw_df.columns).index(required["channel"]) if required["channel"] in raw_df.columns else 0)
        if st.button("Load Uploaded Dataset"):
            cleaned = standardize_feedback_frame(
                raw_df,
                text_column=text_col,
                date_column=date_col,
                store_column=store_col,
                region_column=region_col,
                channel_column=channel_col,
            )
            # Ground truth = which label columns the official file actually provides.
            set_ground_truth(
                issue_type="issue_category" in raw_df.columns,
                severity="severity" in raw_df.columns,
                actionability="actionable" in raw_df.columns,
            )
            analyzer = default_analyzer()
            analyzer.fit_from_dataframe(cleaned)
            missing_targets = [
                column
                for column in ["issue_category", "severity", "actionable", "department"]
                if column not in cleaned.columns
            ]
            if missing_targets:
                with st.spinner("Classifying uploaded records..."):
                    predictions = cleaned["feedback_text"].astype(str).apply(analyzer.analyze)
                if "issue_category" not in cleaned.columns:
                    cleaned["issue_category"] = [pred.issue_category for pred in predictions]
                if "severity" not in cleaned.columns:
                    cleaned["severity"] = [pred.severity for pred in predictions]
                if "actionable" not in cleaned.columns:
                    cleaned["actionable"] = [pred.actionable for pred in predictions]
                if "department" not in cleaned.columns:
                    cleaned["department"] = [pred.department for pred in predictions]
            set_active_dataframe(cleaned, "Uploaded Official Dataset")
            st.success("Uploaded dataset loaded successfully.")
            st.rerun()

    st.markdown("#### Demo Mode")
    if st.button("Reset to Synthetic Demonstration Data"):
        demo = load_demo_bundle()
        set_active_dataframe(demo, "Synthetic Demonstration Data")
        set_ground_truth(issue_type=True, severity=True, actionability=True)
        st.session_state.pop("labeled_training_dataframe", None)
        st.success("Synthetic demo dataset restored.")
        st.rerun()

    st.markdown("#### Local Storage")
    st.write(f"SQLite cache: `{Path('data/storepulse.sqlite')}`")
    st.write(f"Synthetic CSV: `{CSV_PATH}`")
    st.write("All generated demo data is clearly marked as synthetic demonstration data.")
    return df



def app() -> None:
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption(APP_TAGLINE)
    logo_path = Path("assets") / "logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=110)

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Dashboard"

    page = st.sidebar.radio("Navigation", PAGES, index=PAGES.index(st.session_state["selected_page"]))
    st.session_state["selected_page"] = page

    df = prepare_active_dataframe()
    if "feedback_text" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "feedback_text"})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.loc[~df["timestamp"].isna()].copy()
    df["actionable"] = _as_bool_series(df["actionable"]) if "actionable" in df.columns else False
    if not df.empty:
        df["week_start"] = pd.to_datetime(df["timestamp"]).dt.to_period("W-MON").dt.start_time
    else:
        df["week_start"] = pd.Series(dtype="datetime64[ns]")

    labeled_df = st.session_state.get("labeled_training_dataframe")
    analyzer_signature = (
        f"{len(df)}|"
        f"{df['timestamp'].max() if not df.empty else 'empty'}|"
        f"{len(labeled_df) if isinstance(labeled_df, pd.DataFrame) else 0}"
    )
    analyzer = get_analyzer(analyzer_signature)
    analyzer.set_use_embeddings(bool(st.session_state.get("use_embeddings", True)))
    st.session_state["analyzer_signature"] = analyzer_signature

    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert Thresholds")
    minimum_support = int(st.sidebar.slider("Minimum support", 1, 8, 2, help="Minimum complaints needed before an alert can trigger."))
    growth_threshold = float(st.sidebar.slider("Growth threshold", 1.0, 5.0, 2.5, 0.1, help="Complaint rate growth versus historical baseline."))
    spike_threshold = float(st.sidebar.slider("Spike threshold", 1.0, 6.0, 3.0, 0.1, help="Robust spike score based on median and MAD."))
    st.sidebar.caption("Policy-first safety categories always escalate to HIGH severity.")

    st.sidebar.markdown("---")
    st.sidebar.info(f"Active source: {st.session_state.get('active_source', 'Synthetic Demonstration Data')}")
    st.sidebar.caption("Demo mode uses actual synthetic records with seeded incidents.")

    if page == "Dashboard":
        render_dashboard(df, analyzer, minimum_support, growth_threshold, spike_threshold)
    elif page == "AI Feedback Analyzer":
        render_analyzer(analyzer)
    elif page == "Emerging Alerts":
        render_alerts(df, minimum_support, growth_threshold, spike_threshold)
    elif page == "Weekly Regional Reports":
        alerts = compute_emerging_alerts(df, minimum_support, growth_threshold, spike_threshold)
        render_reports(df, alerts)
    elif page == "Store Analytics":
        render_store_analytics(df)
    elif page == "Evaluation & Model Insights":
        render_evaluation(df, minimum_support, growth_threshold, spike_threshold)
    else:
        render_settings(df)


if __name__ == "__main__":
    app()
