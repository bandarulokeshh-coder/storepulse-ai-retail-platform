#!/usr/bin/env python3
"""Streamlit demo for R5 Voice-of-Store Engine with official dataset."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add FULL_CODE/r5 to path for imports
sys.path.insert(0, str(Path(__file__).parent / "FULL_CODE" / "r5"))

from dataio import load_pack
from classifier import VoiceOfStoreClassifier

st.set_page_config(page_title="R5 Voice-of-Store Engine", layout="wide", page_icon="🛍️")

st.markdown("""
<style>
.hero { padding: 2rem; border-radius: 12px; background: linear-gradient(135deg, #2563EB, #1e40af); color: white; }
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin: 0.5rem 0 0; opacity: 0.95; }
.metric-card { border-radius: 12px; padding: 1.2rem; background: #f8fafc; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_official_data():
    """Load official R5 dataset."""
    data_folder = Path(__file__).parent / "R5_Voice_of_Store_Engine" / "data"
    feedback, labels_train, taxonomy, stores, weeks = load_pack(str(data_folder))
    return feedback, labels_train, taxonomy, stores, weeks

@st.cache_resource
def train_classifier():
    """Train classifier on official training data."""
    feedback, labels_train, taxonomy, stores, weeks = load_official_data()
    clf = VoiceOfStoreClassifier()
    clf.train(labels_train, feedback, taxonomy)
    return clf

def main():
    st.markdown("""
    <div class="hero">
      <h1>🛍️ R5 Voice-of-Store Engine</h1>
      <p>Real-time retail feedback classification and incident detection across 60 stores</p>
    </div>
    """, unsafe_allow_html=True)

    feedback, labels_train, taxonomy, stores, weeks = load_official_data()

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Feedback", f"{len(feedback):,}")
    with col2:
        st.metric("Training Records", f"{len(labels_train):,}")
    with col3:
        st.metric("Stores", feedback["store_id"].nunique())
    with col4:
        st.metric("Weeks", len(weeks))
    with col5:
        st.metric("Issue Types", len(taxonomy))

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Analyzer", "📈 Predictions"])

    with tab1:
        st.subheader("Dataset Overview")

        # Date range
        feedback['created_at'] = pd.to_datetime(feedback['created_at'])
        min_date = feedback['created_at'].min().date()
        max_date = feedback['created_at'].max().date()
        st.write(f"**Date Range:** {min_date} to {max_date}")

        # Store distribution
        fig_stores = px.bar(
            feedback['store_id'].value_counts().head(15),
            title="Top 15 Stores by Feedback Volume",
            labels={'value': 'Feedback Count', 'index': 'Store ID'}
        )
        st.plotly_chart(fig_stores, use_container_width=True)

        # Weekly trend
        weekly = feedback.set_index('created_at').resample('W').size()
        fig_trend = px.line(
            x=weekly.index, y=weekly.values,
            title="Weekly Feedback Trend",
            labels={'x': 'Week', 'y': 'Feedback Count'},
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        st.subheader("Test Classifier")
        st.write("Enter feedback text to see real-time classification:")

        clf = train_classifier()

        samples = [
            "Freezer not working, milk packets spoiling!",
            "Wrong allergen label on food packet.",
            "Customer praise for friendly staff.",
        ]

        col1, col2, col3 = st.columns(3)
        for i, sample in enumerate(samples):
            with [col1, col2, col3][i]:
                if st.button(f"Load Sample {i+1}"):
                    st.session_state['test_text'] = sample

        text = st.text_area(
            "Feedback text",
            value=st.session_state.get('test_text', ''),
            height=100
        )
        st.session_state['test_text'] = text

        if text and st.button("Classify"):
            with st.spinner("Classifying..."):
                issues, severities, actionables = clf.predict([text])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Issue Type", issues[0])
            with col2:
                st.metric("Severity", severities[0])
            with col3:
                st.metric("Actionable", "Yes" if actionables[0] else "No")

    with tab3:
        st.subheader("Submission Files")

        pred_path = Path(__file__).parent / "submission" / "R5_predictions.csv"
        flags_path = Path(__file__).parent / "submission" / "R5_emerging_flags.csv"

        if pred_path.exists():
            preds = pd.read_csv(pred_path)
            st.write(f"✅ **Predictions:** {len(preds):,} records generated")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Issue Types", preds['issue_type_id'].nunique())
            with col2:
                actionable_pct = (preds['actionable'].sum() / len(preds) * 100)
                st.metric("Actionable %", f"{actionable_pct:.1f}%")
            with col3:
                high_sev = (preds['severity'] == 'high').sum()
                st.metric("High Severity", high_sev)

            st.dataframe(preds.head(100), use_container_width=True)

        if flags_path.exists():
            flags = pd.read_csv(flags_path)
            st.write(f"✅ **Emerging Flags:** {len(flags):,} incidents detected")
            st.dataframe(flags, use_container_width=True)

    st.markdown("---")
    st.caption("TalentAP Mega Hackathon 2026 — Problem R5 Voice-of-Store Engine")

if __name__ == "__main__":
    main()
