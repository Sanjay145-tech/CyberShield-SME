# the unanswered metric only looks for the exact refusal sentence.
# this reads the saved answers and sorts them into 4 kinds so we can see what the metric misses.
# no ollama needed. run from the repo root:  python eval/refusal_audit.py
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config

VARIANTS = ["llm_only", "bm25", "dense", "hybrid", "bm25_scenario", "hybrid_scenario"]
SHORT = 30   # under this many words = just a refusal, nothing else

def kind(answer):
	words = len(answer.split())
	flagged = config.NO_ANSWER.lower() in answer.lower()
	reworded = ("can't find official acsc guidance" in answer.lower()) and not flagged
	if flagged and words <= SHORT:
		return "clean refusal"
	if flagged:
		return "answer plus refusal sentence"
	if reworded and words <= SHORT + 15:
		return "reworded refusal (metric misses it)"
	return "normal answer"

lines = []
for v in VARIANTS:
	path = ROOT / "results" / "answers_{}.jsonl".format(v)
	if not path.exists():
		continue
	counts = {}
	for line in open(path, encoding="utf-8"):
		r = json.loads(line)
		k = kind(r["answer"])
		counts[k] = counts.get(k, 0) + 1
		if k not in ("normal answer", "clean refusal"):
			lines.append("{},{},{}".format(v, r["qid"], k))
	print(v, counts)

out = ROOT / "results" / "refusal_audit.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", out)
