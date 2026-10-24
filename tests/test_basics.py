# small checks for the parts that do not need ollama or the embedding model
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.incident import classify_query
from src.rag import parse_citations

def test_classify_ransomware():
	assert "ransomware" in classify_query("our files are encrypted and there is a ransom note")

def test_classify_bec():
	assert "bec" in classify_query("supplier changed their bank details")

def test_classify_nothing():
	assert classify_query("how do i register a trademark") == []

def test_citations_valid():
	cited, invalid = parse_citations("Do this. [S1] Then this. [S2][S3]", 5)
	assert cited == [1, 2, 3] and invalid == []

def test_citations_invalid():
	cited, invalid = parse_citations("Do this. [S7]", 5)
	assert 7 in invalid

if __name__ == "__main__":
	for name, fn in list(globals().items()):
		if name.startswith("test_"):
			fn()
			print("ok", name)
