"""Step 2: find the chunks most relevant to a question.

Four retrievers, so the evaluation can compare them:
  bm25      keyword matching (good with exact cyber terms like "MFA", "BEC")
  dense     meaning-based search with a small sentence-embedding model
  hybrid    both, merged with Reciprocal Rank Fusion (RRF)
  + scenario-aware: any of the above, but chunks tagged with the question's
    incident type get boosted (see rag.py / scenario_rerank below)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config

STOPWORDS = set("""a an the and or but if of to in on at for with by from is are was were
be been being it its this that these those i we you they he she my our your their me us
do does did have has had what when where which who whom how why can could should would
will just so not no than then there here about into as up out""".split())


def tokenize(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def load_chunks():
    if not config.CHUNKS_FILE.exists():
        sys.exit("No chunks found. Run `python src/ingest.py` first.")
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class BM25Retriever:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"] + " " + c["title"]) for c in chunks])

    def scores(self, query):
        return self.bm25.get_scores(tokenize(query))


class DenseRetriever:
    def __init__(self, chunks):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.model = SentenceTransformer(config.EMBED_MODEL)
        if config.EMB_FILE.exists():
            emb = np.load(config.EMB_FILE)
            if emb.shape[0] == len(chunks):
                self.emb = emb
                return
        print("Embedding chunks (first run only)...")
        self.emb = self.model.encode([c["text"] for c in chunks], batch_size=32,
                                     normalize_embeddings=True, show_progress_bar=True)
        np.save(config.EMB_FILE, self.emb)

    def scores(self, query):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return self.emb @ q


def rrf(score_lists, k=60):
    """Reciprocal Rank Fusion: combine rankings without tuning score scales."""
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        ranks = np.argsort(-np.asarray(scores))
        for r, idx in enumerate(ranks):
            fused[idx] += 1.0 / (k + r + 1)
    return fused


def scenario_rerank(scores, chunks, incident_types, boost=1.5):
    """Boost chunks tagged with the question's incident type(s).

    A boost (not a hard filter) is used on purpose: if the classifier is wrong,
    strong general matches can still win, which limits the damage of
    misclassifying overlapping incidents.
    """
    if not incident_types:
        return scores
    scores = np.asarray(scores, dtype=float).copy()
    # shift to non-negative so multiplying always helps
    scores = scores - scores.min() + 1e-9
    for i, c in enumerate(chunks):
        if set(c["incident_tags"]) & set(incident_types):
            scores[i] *= boost
    return scores


class Retriever:
    """One object the rest of the code uses. method: bm25 | dense | hybrid."""

    def __init__(self, method="hybrid"):
        self.method = method
        self.chunks = load_chunks()
        self.bm25 = BM25Retriever(self.chunks) if method in ("bm25", "hybrid") else None
        self.dense = DenseRetriever(self.chunks) if method in ("dense", "hybrid") else None

    def search(self, query, k=config.TOP_K, incident_types=None):
        if self.method == "bm25":
            scores = self.bm25.scores(query)
        elif self.method == "dense":
            scores = self.dense.scores(query)
        else:
            scores = rrf([self.bm25.scores(query), self.dense.scores(query)])
        if incident_types:
            scores = scenario_rerank(scores, self.chunks, incident_types)
        top = np.argsort(-np.asarray(scores))[:k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in top]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "supplier changed bank details by email"
    r = Retriever("bm25")
    for hit in r.search(q):
        print(f"{hit['score']:.3f}  {hit['source']} p{hit['page']}  {hit['text'][:100]}...")
