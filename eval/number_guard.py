# checks every phone number in the saved answers against the ACSC documents.
# no ollama needed, it only reads files that already exist.
# run from the repo root:  python eval/number_guard.py
import sys
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["llm_only", "bm25", "dense", "hybrid", "bm25_scenario", "hybrid_scenario"]

# phone numbers like 1300 292 371 or 13 00 or (03) 9925 1234
pattern = re.compile(r"1[38]00[\s]?[\d\s]{6,9}|\b13[\s]?\d{2}[\s]?\d{2}\b")

def clean(text):
	return re.sub(r"\D", "", text)

def numbers_in(text):
	found = []
	for m in pattern.finditer(text):
		digits = clean(m.group(0))
		if len(digits) >= 6:
			found.append(digits)
	return found

# every number that appears anywhere in the 11 ACSC documents
corpus_numbers = set()
chunks_file = ROOT / "data" / "processed" / "chunks.jsonl"
with open(chunks_file, encoding="utf-8") as f:
	for line in f:
		corpus_numbers.update(numbers_in(json.loads(line)["text"]))
# the hotline is written as 1300 CYBER1 in the guides, so add its digits
corpus_numbers.add("1300292371")

flagged = []
for v in VARIANTS:
	path = ROOT / "results" / "answers_{}.jsonl".format(v)
	if not path.exists():
		continue
	with open(path, encoding="utf-8") as f:
		for line in f:
			r = json.loads(line)
			for n in numbers_in(r["answer"]):
				# for rag answers the number must be in a retrieved chunk
				if r["hits"]:
					source_text = " ".join(h["text"] for h in r["hits"])
					ok = n in [clean(x) for x in numbers_in(source_text)] or n == "1300292371"
				else:
					ok = n in corpus_numbers
				if not ok:
					flagged.append((v, r["qid"], n))

print("answers with a phone number not found in the sources:")
for v, q, n in flagged:
	print("  {:16} {:5} {}".format(v, q, n))
print("total flagged: {}".format(len(flagged)))

out = ROOT / "results" / "number_guard_report.txt"
with open(out, "w", encoding="utf-8") as f:
	for v, q, n in flagged:
		f.write("{},{},{}\n".format(v, q, n))
print("saved to results/number_guard_report.txt")
