import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA

st.set_page_config(page_title="NLP Text Similarity", layout="wide")

st.title("🧠 Text / Word Similarity using Pretrained NLP Model")
st.markdown("**Model used:** `all-MiniLM-L6-v2` from Sentence Transformers (free, no training required)")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.markdown("---")
st.header("📝 Enter Your Text")

default_texts = "king\nqueen\nman\nwoman\nprince\nprincess\ndoctor\nnurse"
user_input = st.text_area(
    "Enter words or sentences (one per line):",
    value=default_texts,
    height=180
)

if st.button("🔍 Compute Similarity"):
    items = [line.strip() for line in user_input.strip().split("\n") if line.strip()]

    if len(items) < 2:
        st.warning("Please enter at least 2 words or sentences.")
    else:
        embeddings = model.encode(items, convert_to_tensor=True)
        cos_sim_matrix = util.cos_sim(embeddings, embeddings).numpy()

        st.markdown("---")
        st.header("📊 Similarity Results")

        # --- Top Similar Pairs ---
        st.subheader("Top Similar Pairs")
        pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pairs.append((items[i], items[j], float(cos_sim_matrix[i][j])))
        pairs.sort(key=lambda x: x[2], reverse=True)

        top_pairs = pairs[:min(8, len(pairs))]
        for p in top_pairs:
            st.write(f"**{p[0]}** ↔ **{p[1]}** → Score: `{p[2]:.4f}`")

        col1, col2, col3 = st.columns(3)

        # --- Graph 1: Bar Chart ---
        with col1:
            st.subheader("📊 Bar Chart — Top Similar Pairs")
            labels = [f"{p[0]} & {p[1]}" for p in top_pairs]
            scores = [p[2] for p in top_pairs]

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            bars = ax1.barh(labels[::-1], scores[::-1], color="steelblue")
            ax1.set_xlabel("Cosine Similarity Score")
            ax1.set_title("Top Similar Pairs")
            ax1.set_xlim(0, 1)
            for bar, score in zip(bars, scores[::-1]):
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{score:.3f}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig1)

        # --- Graph 2: Heatmap ---
        with col2:
            st.subheader("🔥 Heatmap — Pairwise Similarity")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cos_sim_matrix,
                annot=True, fmt=".2f",
                xticklabels=items, yticklabels=items,
                cmap="YlOrRd", ax=ax2,
                linewidths=0.5
            )
            ax2.set_title("Pairwise Cosine Similarity")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)

        # --- Graph 3: 2D PCA Plot ---
        with col3:
            st.subheader("🗺️ 2D PCA Embedding Plot")
            emb_np = embeddings.numpy()
            n_components = min(2, len(items))
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(emb_np)

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.scatter(reduced[:, 0], reduced[:, 1] if n_components > 1 else np.zeros(len(items)),
                        color="darkorange", s=80, zorder=5)
            for i, label in enumerate(items):
                ax3.annotate(label,
                             (reduced[i, 0], reduced[i, 1] if n_components > 1 else 0),
                             textcoords="offset points", xytext=(6, 4), fontsize=9)
            ax3.set_title("PCA 2D Projection of Embeddings")
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            plt.tight_layout()
            st.pyplot(fig3)

        # --- Paul's Critical Thinking Standards ---
        st.markdown("---")
        st.header("🧠 Paul's Critical Thinking Standards")

        best = top_pairs[0]

        standards = {
            "✅ Clarity": f"The user entered {len(items)} items. The model computed cosine similarity scores between all pairs. A score of 1.0 means identical, 0.0 means unrelated.",
            "✅ Accuracy": f"Model used: `all-MiniLM-L6-v2` (Sentence Transformers, HuggingFace). No custom training or preprocessing was performed.",
            "✅ Precision": f"The highest similarity score is `{best[2]:.4f}` between **{best[0]}** and **{best[1]}**. Exact numeric scores are shown for all pairs.",
            "✅ Relevance": "All three graphs directly support the similarity results: bar chart shows top pairs, heatmap shows all pair scores, and PCA shows spatial relationships.",
            "✅ Logic": f"**{best[0]}** and **{best[1]}** have the highest score of `{best[2]:.4f}`, which logically makes sense as they likely share semantic meaning in the model's embedding space.",
            "✅ Significance": f"The most significant result is the pair **{best[0]}** ↔ **{best[1]}** with score `{best[2]:.4f}`, which represents the strongest semantic connection in the input.",
            "⚠️ Fairness (Limitation)": "The `all-MiniLM-L6-v2` model was trained primarily on English text. It may not accurately capture similarity for domain-specific jargon, non-English words, or very short single characters.",
        }

        for standard, explanation in standards.items():
            st.markdown(f"**{standard}:** {explanation}")

st.markdown("---")
st.caption("NLP Lab Quiz | Shifa Tameer-e-Millat University | Model: all-MiniLM-L6-v2 | Free & No Training")
