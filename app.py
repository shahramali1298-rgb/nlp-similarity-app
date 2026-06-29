import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import pipeline

st.set_page_config(page_title="NLP - Paul's Critical Thinking Standards", layout="wide")

st.title("🧠 Paul's Critical Thinking Standards Analyzer")
st.markdown("**Model:** `monologg/bert-base-cased-goemotions-original` (Google GoEmotions — Free, No Training)")
st.markdown("---")

# ── Load GoEmotions model ──
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="monologg/bert-base-cased-goemotions-original",
        top_k=None
    )

classifier = load_model()

# ── Mapping: GoEmotions → Paul's 7 Standards ──
STANDARDS_MAP = {
    "Clarity":      ["confusion", "curiosity", "realization", "surprise"],
    "Accuracy":     ["approval", "disapproval", "admiration", "disappointment"],
    "Precision":    ["nervousness", "annoyance", "desire", "caring"],
    "Relevance":    ["excitement", "amusement", "gratitude", "relief"],
    "Logic":        ["pride", "optimism", "remorse", "embarrassment"],
    "Significance": ["joy", "love", "sadness", "fear"],
    "Fairness":     ["anger", "disgust", "grief", "neutral"],
}

def compute_standards(text):
    results = classifier(text)[0]
    emotion_scores = {r["label"]: r["score"] for r in results}

    standards_scores = {}
    for standard, emotions in STANDARDS_MAP.items():
        scores = [emotion_scores.get(e, 0.0) for e in emotions]
        standards_scores[standard] = round(np.mean(scores) * 100, 2)

    # Normalize to 0-100 range nicely
    total = sum(standards_scores.values())
    if total > 0:
        standards_scores = {k: round(v / total * 100, 2) for k, v in standards_scores.items()}

    return standards_scores, emotion_scores

# ── UI ──
st.header("📝 Enter Your Text")
user_text = st.text_area(
    "Type any sentence or paragraph:",
    value="The doctor carefully explained the diagnosis with clear evidence and logical reasoning.",
    height=130
)

if st.button("🔍 Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            standards_scores, emotion_scores = compute_standards(user_text)

        st.markdown("---")
        st.header("📊 Paul's Critical Thinking Standards — Scores")

        standards = list(standards_scores.keys())
        scores    = list(standards_scores.values())
        colors    = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3"]

        col1, col2, col3 = st.columns(3)

        # ── Graph 1: Horizontal Bar Chart ──
        with col1:
            st.subheader("📊 Standards Score Bar Chart")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            bars = ax1.barh(standards[::-1], scores[::-1], color=colors[::-1])
            ax1.set_xlabel("Score (%)")
            ax1.set_xlim(0, max(scores) * 1.25)
            ax1.set_title("Paul's Standards Scores")
            for bar, score in zip(bars, scores[::-1]):
                ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                         f"{score:.1f}%", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig1)

        # ── Graph 2: Pie Chart ──
        with col2:
            st.subheader("🥧 Standards Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            wedges, texts, autotexts = ax2.pie(
                scores, labels=standards, autopct="%1.1f%%",
                colors=colors, startangle=140,
                textprops={"fontsize": 8}
            )
            ax2.set_title("% Distribution of Standards")
            plt.tight_layout()
            st.pyplot(fig2)

        # ── Graph 3: Radar Chart ──
        with col3:
            st.subheader("🕸️ Radar Chart")
            N = len(standards)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            values = scores + [scores[0]]
            angles += angles[:1]

            fig3, ax3 = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
            ax3.plot(angles, values, color="#4C72B0", linewidth=2)
            ax3.fill(angles, values, color="#4C72B0", alpha=0.25)
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(standards, fontsize=8)
            ax3.set_title("Standards Radar", pad=15)
            plt.tight_layout()
            st.pyplot(fig3)

        # ── Scores Table ──
        st.markdown("---")
        st.subheader("📋 Exact Scores")
        cols = st.columns(7)
        for i, (std, score) in enumerate(standards_scores.items()):
            with cols[i]:
                st.metric(label=std, value=f"{score:.1f}%")

        # ── Top & Lowest Standard ──
        top_std = max(standards_scores, key=standards_scores.get)
        low_std = min(standards_scores, key=standards_scores.get)
        st.success(f"✅ **Highest:** {top_std} — {standards_scores[top_std]:.1f}%")
        st.error(f"⚠️ **Lowest:** {low_std} — {standards_scores[low_std]:.1f}%")

st.markdown("---")
st.caption("NLP Lab Quiz | Shifa Tameer-e-Millat University | Google GoEmotions Model | Free & No Training")
