# agreement between the two human raters (anitha and karthik).
# the sheets use all / not all for covered_count, so agreement.py can't read them.
# no ollama needed. run from the repo root:  python eval/human_agreement.py
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def load(name):
	rows = {}
	f = open(ROOT / "results" / name, encoding="utf-8", newline="")
	for r in list(csv.reader(f))[1:]:
		if len(r) >= 8 and r[0].strip().isdigit():
			rows[int(r[0])] = (r[-3].strip(), r[-2].strip(), r[-1].strip())
	f.close()
	return rows

def flag(rows, col):
	# 1 if the cell starts with 1 (so "1ig" counts as 1), items marked ig are unsure
	out = {}
	for i in rows:
		v = rows[i][col]
		if v == "ig":
			out[i] = None
		elif col == 0:
			out[i] = 1 if v == "not all" else 0
		else:
			out[i] = 1 if v.startswith("1") else 0
	return out

def kappa(a, b):
	items = [i for i in a if a[i] is not None and b[i] is not None]
	n = len(items)
	po = sum(1 for i in items if a[i] == b[i]) / n
	pa = sum(a[i] for i in items) / n
	pb = sum(b[i] for i in items) / n
	pe = pa * pb + (1 - pa) * (1 - pb)
	if pe == 1:
		return n, po, 1.0
	return n, po, (po - pe) / (1 - pe)

A = load("human_rating_anitha.csv")
K = load("human_rating_karthik.csv")
names = ["not all key points", "wrong advice", "unsafe"]
for col in range(3):
	fa = flag(A, col)
	fk = flag(K, col)
	n, po, k = kappa(fa, fk)
	print("{}: items={} agree={:.2f} kappa={:.2f}".format(names[col], n, po, k))
	print("   anitha flagged", [i for i in fa if fa[i] == 1], "karthik flagged", [i for i in fk if fk[i] == 1], "unsure", [i for i in fk if fk[i] is None])
