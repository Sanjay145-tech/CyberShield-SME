"""Step 3: the full assistant. Question -> (classify) -> retrieve -> cited answer.

Usage:  python src/rag.py "My supplier emailed new bank details, what do I do?"

Pipeline variants (compared in evaluation):
  llm_only          no retrieval at all (the "just ask ChatGPT" baseline)
  bm25 / dense / hybrid           standard RAG
  *_scenario        classify the incident type first, then retrieve
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from incident import classify_query
from llm import OllamaLLM
from retrieval import Retriever

RAG_SYSTEM = f"""You are CyberShield SME, an assistant that helps Australian small
business owners respond to cyber incidents and scams. Your readers are not IT experts.

Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. After every sentence that uses a source, cite it like [S1] or [S2][S3].
3. Give clear numbered steps, most urgent first. Plain language, no jargon.
4. Never invent phone numbers, websites, or organisations that are not in the sources.
5. If the user asks whether they should do something risky (pay a ransom, pay a
   changed bank account, confirm details by replying to the email, approve an MFA
   prompt they did not start, turn off MFA), ANSWER the question: say clearly not
   to, explain why, and give the safe alternative, all from the sources.
6. If the sources say who to report to or contact, include that step.
7. Give the complete answer now. Do not promise further help or follow-up steps.
8. Only if the sources contain nothing relevant to the question, reply with exactly:
   "{config.NO_ANSWER}" and nothing else."""

# Same task and refusal rule as RAG, just no sources: a fair baseline.
LLM_ONLY_SYSTEM = f"""You are an assistant that helps Australian small business owners
respond to cyber incidents and scams. Give clear numbered steps in plain language.
If you are not confident you can answer, reply with exactly:
{config.NO_ANSWER}"""


def format_sources(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[S{i}] ({h['title']}, page {h['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def parse_citations(answer, n_sources):
    """Return the source numbers cited, plus any that point to nothing (invalid)."""
    cited = sorted({int(x) for x in re.findall(r"\[S(\d+)\]", answer)})
    valid = [c for c in cited if 1 <= c <= n_sources]
    invalid = [c for c in cited if c not in valid]
    return valid, invalid


class CyberShield:
    def __init__(self, variant="hybrid_scenario", llm=None):
        self.variant = variant
        self.llm = llm or OllamaLLM()
        self.scenario = variant.endswith("_scenario")
        method = variant.replace("_scenario", "")
        self.retriever = None if method == "llm_only" else Retriever(method)

    def answer(self, question: str) -> dict:
        if self.retriever is None:
            text = self.llm.chat(LLM_ONLY_SYSTEM, question)
            return {"question": question, "variant": self.variant, "answer": text,
                    "incident_types": [], "hits": [], "cited": [], "invalid_citations": [],
                    "unanswered": config.NO_ANSWER.lower() in text.lower()}

        incident_types = classify_query(question) if self.scenario else []
        hits = self.retriever.search(question, incident_types=incident_types)
        prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
        text = self.llm.chat(RAG_SYSTEM, prompt)
        cited, invalid = parse_citations(text, len(hits))
        return {
            "question": question,
            "variant": self.variant,
            "answer": text,
            "incident_types": incident_types,
            "hits": hits,
            "cited": cited,                  # source numbers, 1-based
            "invalid_citations": invalid,    # cited numbers that don't exist
            "unanswered": config.NO_ANSWER.lower() in text.lower(),
        }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "My supplier emailed saying their bank details changed. Should I pay?"
    out = CyberShield("hybrid_scenario").answer(q)
    print("Incident type:", out["incident_types"] or "unclassified")
    print("\n" + out["answer"] + "\n")
    for i in out["cited"]:
        h = out["hits"][i - 1]
        print(f"[S{i}] {h['title']}, page {h['page']}  ({h['source']})")
