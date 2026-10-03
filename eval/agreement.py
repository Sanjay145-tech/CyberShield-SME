"""Step 5: check the LLM judge against humans (this is the "validated" part).

Two team members fill in results/human_rating_sheet.csv independently
(no peeking at each other or at human_rating_key.csv). Then run:
  python eval/agreement.py

Reports Cohen's kappa for rater1 vs rater2 and for human (rater1) vs LLM judge.
Rough reading: >0.6 substantial, 0.4-0.6 moderate, <0.4 weak (don't trust the
judge on that metric; report human scores instead).
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def main():
    sheet = RES / "human_rating_sheet.csv"
    key = RES / "human_rating_key.csv"
    if not sheet.exists():
        sys.exit("Run eval/run_eval.py first to create the rating sheet.")
    with open(sheet, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["rater1_unsafe(0/1)"].strip()]
    with open(key, encoding="utf-8") as f:
        keys = {r["item"]: r for r in csv.DictReader(f)}
    if not rows:
        sys.exit("No ratings filled in yet.")

    print(f"{len(rows)} rated items\n")
    print(f"{'measure':<12}{'rater1 vs rater2':>18}{'human vs judge':>17}")
    for field, jfield in [("incorrect", "judge_incorrect"), ("unsafe", "judge_unsafe")]:
        r1 = [int(r[f"rater1_{field}(0/1)"]) for r in rows]
        r2 = [int(r[f"rater2_{field}(0/1)"] or -1) for r in rows]
        jd = [int(keys[r["item"]][jfield]) for r in rows]
        both = [(x, y) for x, y in zip(r1, r2) if y != -1]
        k12 = kappa([x for x, _ in both], [y for _, y in both]) if both else None
        print(f"{field:<12}{str(k12):>18}{str(kappa(r1, jd)):>17}")

    # key-point coverage: count agreement within +/-1 point
    diffs = [abs(int(r["rater1_covered_count"]) - int(keys[r["item"]]["judge_covered_count"]))
             for r in rows if r["rater1_covered_count"].strip()]
    if diffs:
        within = sum(d <= 1 for d in diffs) / len(diffs)
        print(f"\nKey-point coverage: judge within 1 point of rater1 on {within:.0%} of items")


if __name__ == "__main__":
    main()
