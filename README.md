# Maths-RAG-Project

Class 12 CBSE Mathematics RAG Agent — powered by NCERT notes, exemplar, important questions, PYQs, competency-based questions, and secret assignments.

## Run locally

```bash
pip install -r requirements.txt
python -m math_tutor.ingest        # build vector store (one-time)
streamlit run math_tutor/app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `math_tutor/app.py`
4. Add your secrets under **Settings → Secrets**:

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "gsk_..."
```
