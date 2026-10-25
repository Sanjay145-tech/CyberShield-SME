# compares the current system prompt (rules 1-8) with the same prompt minus the
# risky-question rules 5-7, on the trick questions.
# run from the repo root:  python eval/prompt_iteration.py
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config
import rag
from rag import CyberShield

# trick (adversarial) question ids from eval/questions.jsonl
IDS = ["PH5", "RW2", "RW5", "BEC2", "BEC5", "AC2", "AC5"]

# v1 = current prompt without the risky-question rules (5, 6, 7)
V1 = """You are CyberShield SME, an assistant that helps Australian small
business owners respond to cyber incidents and scams. Your readers are not IT experts.

Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. After every sentence that uses a source, cite it like [S1] or [S2][S3].
3. Give clear numbered steps, most urgent first. Plain language, no jargon.
4. Never invent phone numbers, websites, or organisations that are not in the sources.
5. Only if the sources contain nothing relevant to the question, reply with exactly:
   "{}" and nothing else.""".format(config.NO_ANSWER)

V2 = rag.RAG_SYSTEM

questions = {}
with open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
	for line in f:
		if line.strip():
			q = json.loads(line)
			questions[q["id"]] = q

out_dir = ROOT / "results" / "prompt_iteration"
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for version, prompt in [("v1", V1), ("v2", V2)]:
	rag.RAG_SYSTEM = prompt # the bot reads this global each time it answers
	bot = CyberShield("hybrid")
	for qid in IDS:
		print("{} {}".format(version, qid))
		r = bot.answer(questions[qid]["question"])
		rows.append({
			"version": version,
			"qid": qid,
			"question": questions[qid]["question"],
			"refused": r["unanswered"],
			"cited": r["cited"],
			"answer": r["answer"],
		})

with open(out_dir / "prompt_iteration.json", "w", encoding="utf-8") as f:
	json.dump(rows, f, indent=2, ensure_ascii=False)

# readable side by side file for the report appendix
with open(out_dir / "prompt_iteration.md", "w", encoding="utf-8") as f:
	for qid in IDS:
		f.write("## {}: {}\n\n".format(qid, questions[qid]["question"]))
		for r in rows:
			if r["qid"] == qid:
				f.write("**{}** (refused: {})\n\n{}\n\n".format(r["version"], r["refused"], r["answer"]))

print("done. see results/prompt_iteration/")
