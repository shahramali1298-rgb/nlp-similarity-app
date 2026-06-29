# NLP Text Similarity App

**NLP Lab Quiz — Shifa Tameer-e-Millat University, Islamabad**

## Model Used
`all-MiniLM-L6-v2` — a free pretrained sentence embedding model from [HuggingFace Sentence Transformers](https://www.sbert.net/).  
No training, no preprocessing, no paid API used.

## App Purpose
This Streamlit app computes **text/word similarity** using cosine similarity on pretrained embeddings.  
Users enter words or sentences, and the app shows:
- Similarity scores between all pairs
- Bar chart of top similar pairs
- Heatmap of pairwise similarity
- 2D PCA embedding plot
- Paul's Critical Thinking Standards analysis

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Deployed App
[Click here to open the app](#) ← replace with your Streamlit Cloud link after deploying

## Screenshots
*(Add screenshots of your running app here after deployment)*

## Files
| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Paul's Critical Thinking Standards Applied
| Standard | Implementation |
|----------|---------------|
| Clarity | Explains input/output meaning |
| Accuracy | Shows exact model name |
| Precision | Exact numeric scores shown |
| Relevance | All graphs support results |
| Logic | Explains why top result makes sense |
| Significance | Highlights most important result |
| Fairness | Mentions model limitations |
