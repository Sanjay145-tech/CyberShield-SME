# CyberShield SME (Group 91, Test-Driven RAG)

A RAG assistant that gives Australian small businesses step-by-step responses to
phishing, ransomware, business email compromise (BEC) and account compromise,
grounded only in official ACSC guidance, plus an evaluation framework that
measures whether its advice is correct, complete, faithful, cited and safe.

## Pipeline

```
question -> incident classifier -> retriever (BM25 / dense / hybrid) -> top-5 ACSC chunks
         -> local LLM (Ollama) with strict "sources only, cite [S#]" prompt -> cited answer
```

| File | What it does |
|---|---|
| `src/ingest.py` | PDFs / saved web pages -> overlapping chunks tagged with incident type and page |
| `src/incident.py` | keyword incident classifier (the "scenario-aware" step) |
| `src/retrieval.py` | BM25, dense (MiniLM), hybrid (RRF), incident-type boosting |
| `src/rag.py` | full assistant + `llm_only` baseline (no sources) |
| `eval/questions.jsonl` | 24 test questions: 20 answerable (7 adversarial), 4 out-of-scope |
| `eval/run_eval.py` | runs every variant, LLM judge, metrics table, chart, human rating sheet |
| `eval/agreement.py` | Cohen's kappa: rater vs rater and human vs LLM judge |
| `app.py` | Streamlit demo, with side-by-side "plain AI vs CyberShield" mode |

## Setup (Python 3.10+)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b      # weak laptop: ollama pull llama3.2:3b and edit src/config.py
```

## Run

1. Download the ACSC documents (PDF, or save web pages as .html) into `data/raw/`.
   Record each one in `data/sources.csv` (filename, title, url, date accessed).
   The date column is how we handle "ACSC updates its guidance".
2. `python src/ingest.py`
3. `python src/rag.py "My supplier emailed new bank details. Should I pay?"`
4. `python eval/run_eval.py --limit 3` (smoke test), then `python eval/run_eval.py`
5. `streamlit run app.py`

Outputs land in `results/`: `summary.md`, `summary.csv`, `summary.png` (video chart),
`human_rating_sheet.csv`.

## Evaluation: what has to be done by people

- **Fill `gold_sources` in `eval/questions.jsonl`** with the filename(s) in `data/raw/`
  that answer each question. Without this, recall@5, MRR and citation-to-gold are blank.
- **Check every `key_points` list against the actual ACSC text.** They are drafts; the
  reference answer must come from the corpus, not from memory.
- **Add more questions.** 24 is a minimum. Paraphrases of existing questions test robustness.
- **Validate the judge.** Two people fill `results/human_rating_sheet.csv` independently
  (variant names are hidden), then run `python eval/agreement.py`. If kappa < 0.4 on a
  metric, report the human scores for it, not the judge's.
- **Caveat on the classifier.** Its keywords were checked against this test set, so its
  accuracy here is optimistic. Write a few new questions it hasn't seen before reporting
  classifier accuracy.

## Variants compared

`llm_only` (plain AI), `bm25`, `dense`, `hybrid`, `bm25_scenario`, `hybrid_scenario` (full system).
