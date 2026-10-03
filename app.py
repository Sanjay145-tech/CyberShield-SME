"""CyberShield SME demo app.   Run:  streamlit run app.py"""
import html
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag import CyberShield  # noqa: E402

st.set_page_config(page_title="CyberShield SME", page_icon="🛡️", layout="wide")

# ---------- design tokens ----------
# palette: Void Black #0B0F14 (page), Neon Cyan #3DF2E0 (accent: hotline, incident marker, citations)
# surfaces #121821 / #18212C, text #E6EDF3, muted #8A97A6, lines #232D3A
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;800&display=swap');
:root { --ink:#E6EDF3; --steel:#3DF2E0; --muted:#8A97A6; --line:#232D3A;
        --page:#0B0F14; --surface:#121821; --raised:#18212C; --signal:#3DF2E0; --ok:#3DF2E0; }

html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important; }
.stApp { background: var(--page); color: var(--ink); }
footer, .stAppDeployButton, [data-testid="stAppDeployButton"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 6rem; }

/* header */
.cs-top { display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem;
          flex-wrap:wrap; border-bottom:2px solid var(--line); padding-bottom:1.1rem; margin-bottom:1.6rem; }
.stMarkdown p.cs-title, p.cs-title { font-size:2.6rem !important; font-weight:800; letter-spacing:-0.02em; line-height:1; margin:0; color:var(--ink); }
.stMarkdown p.cs-sub { color:var(--muted); font-size:1.02rem; margin:.55rem 0 0; max-width:52ch; line-height:1.5; }
.cs-hotline { background:var(--raised); color:var(--ink); padding:.7rem 1rem; border-left:6px solid var(--signal);
              font-size:.9rem; line-height:1.35; }
.cs-hotline b { color:var(--signal); font-size:1.15rem; font-weight:800; display:block; }

/* situation tiles */
.cs-h2 { font-size:1.15rem; font-weight:700; margin:0 0 .6rem; color:var(--ink); }
div.stButton, div[data-testid="stButton"] { width:100%; }
div.stButton > button { width:100%; text-align:left; justify-content:flex-start; background:var(--surface);
  color:var(--ink); border:1px solid var(--line); border-radius:4px; padding:.85rem 1rem; font-weight:500;
  font-size:.98rem; min-height:3.4rem; }
div.stButton > button:hover { border-color:var(--steel); color:var(--ink); background:var(--raised); }
div.stButton > button:focus-visible { outline:3px solid var(--signal); outline-offset:2px; }
div.stButton > button > div, div.stButton > button [data-testid="stMarkdownContainer"] { width:100%; justify-content:flex-start; text-align:left; }
div.stButton > button p { text-align:left; width:100%; }

/* question echo */
.cs-asked { color:var(--muted); font-size:.95rem; margin:1.8rem 0 .5rem; }
.cs-asked b { color:var(--ink); font-weight:600; }

/* answer card: a Streamlit container keyed cscard_* */
[class*="st-key-cscard"] { background:var(--surface); border:1px solid var(--line);
  border-top:6px solid var(--signal); padding:1.2rem 1.4rem 1rem; gap:.4rem; }
.st-key-cscard_plain { border-top-color:#4A5563 !important; }
[class*="st-key-cscard"] p, [class*="st-key-cscard"] li { font-size:1rem; line-height:1.6; max-width:72ch; }
[class*="st-key-cscard"] li { margin-bottom:.4rem; }
.cs-kind { font-weight:700; font-size:1.05rem; margin:0 0 .15rem; color:var(--ink); }
.cs-kindnote { color:var(--muted); font-size:.88rem; margin:0 0 .9rem; }
.cs-cite { display:inline-block; font-size:.72rem; font-weight:600; color:var(--steel);
           border:1px solid #1F5F5A; border-radius:3px; padding:0 .3rem; margin-left:.15rem;
           vertical-align:2px; line-height:1.35; }

/* sources */
.cs-src-h { font-size:.95rem; font-weight:700; margin:1.1rem 0 .4rem; padding-top:.8rem;
            border-top:1px solid var(--line); }
.cs-src { display:grid; grid-template-columns:2.4rem 1fr; gap:.2rem .6rem; padding:.45rem 0;
          font-size:.9rem; line-height:1.45; }
.cs-src-n { font-weight:700; color:var(--steel); }
.cs-src-t { color:var(--ink); font-weight:600; }
.cs-src-t a { color:var(--steel); }
.cs-src-q { grid-column:2; color:var(--muted); }
.cs-tag { font-size:.75rem; font-weight:600; color:var(--ok); margin-left:.4rem; }
.cs-tag.no { color:var(--muted); font-weight:500; }

.cs-colhead { font-weight:700; font-size:1rem; margin:0 0 .5rem; }
.cs-colhead span { color:var(--muted); font-weight:400; }
.cs-foot { color:var(--muted); font-size:.82rem; margin-top:2.5rem; max-width:70ch; line-height:1.5; }

[data-testid="stSidebar"] { background:#080B10; border-right:1px solid var(--line); }
@media (max-width: 640px) { .stMarkdown p.cs-title { font-size:2rem !important; } }
</style>
""", unsafe_allow_html=True)

LABELS = {"phishing": "Phishing", "ransomware": "Ransomware",
          "bec": "Business email compromise", "account_compromise": "Account compromise"}

SITUATIONS = [
    ("A supplier says their bank details changed",
     "A supplier emailed saying their bank details changed. Should I pay the next invoice to the new account?"),
    ("Someone opened a suspicious attachment",
     "An employee opened a suspicious attachment. What do we do right now?"),
    ("I got a sign-in prompt I didn't start",
     "I got an MFA approval request I didn't start. Is my account compromised?"),
    ("Our files are locked with a ransom note",
     "Our files are encrypted and there's a ransom note. What should we do?"),
]


@st.cache_resource(show_spinner="Loading the ACSC guides...")
def load_bot(variant):
    return CyberShield(variant)


def format_answer(text):
    """Escape model output, then turn [S1] markers into small citation tags."""
    safe = html.escape(text, quote=False)
    return re.sub(r"\[S(\d+)\]", r'<span class="cs-cite">S\1</span>', safe)


def render_answer(out, grounded=True):
    if out["unanswered"]:
        kind, note = "No official guidance found", "The ACSC guides in this tool don't cover this question."
    elif out["incident_types"]:
        kind = " and ".join(LABELS.get(t, t) for t in out["incident_types"])
        note = "Steps below come from official ACSC guidance."
    elif grounded:
        kind, note = "Your situation", "Steps below come from official ACSC guidance."
    else:
        kind, note = "General AI answer", "Written without any official sources."

    card = st.container(key=f"cscard_{'grounded' if grounded else 'plain'}")
    card.markdown(f'<p class="cs-kind">{html.escape(kind)}</p><p class="cs-kindnote">{note}</p>',
                  unsafe_allow_html=True)
    card.markdown(format_answer(out["answer"]), unsafe_allow_html=True)

    if grounded and out["hits"] and not out["unanswered"]:
        cited = set(out["cited"])
        rows = ['<p class="cs-src-h">Where this comes from</p>']
        for i, h in enumerate(out["hits"], 1):
            if i not in cited:
                continue
            title = html.escape(h["title"])
            if h.get("url"):
                title = f'<a href="{html.escape(h["url"])}" target="_blank">{title}</a>'
            rows.append(
                f'<div class="cs-src"><span class="cs-src-n">S{i}</span>'
                f'<span class="cs-src-t">{title}, page {h["page"]}</span>'
                f'<span class="cs-src-q">{html.escape(h["text"][:220])}...</span></div>')
        if len(rows) > 1:
            card.markdown("".join(rows), unsafe_allow_html=True)


# ---------- header ----------
st.markdown("""
<div class="cs-top">
  <div>
    <p class="cs-title">CyberShield SME</p>
    <p class="cs-sub">Step-by-step help when your business is hit by a scam or cyber attack,
    taken only from official Australian Cyber Security Centre guidance.</p>
  </div>
  <div class="cs-hotline">ACSC 24/7 Hotline<b>1300 CYBER1</b>1300 292 371</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Demo settings**")
    compare = st.toggle("Compare with a general AI", value=False)
    variant = st.selectbox(
        "Search method", ["hybrid_scenario", "hybrid", "bm25_scenario", "bm25", "dense"],
        format_func=lambda v: {
            "hybrid_scenario": "Hybrid + incident type", "hybrid": "Hybrid",
            "bm25_scenario": "Keyword + incident type", "bm25": "Keyword", "dense": "Meaning"}[v])
    if st.button("Clear answer"):
        st.session_state.pop("question", None)
    st.caption("Student prototype (RMIT, Group 91). Not a replacement for professional incident response.")

# ---------- situations ----------
st.markdown('<p class="cs-h2">What\'s happening?</p>', unsafe_allow_html=True)
cols = st.columns(2)
for i, (label, full) in enumerate(SITUATIONS):
    if cols[i % 2].button(label, key=f"sit{i}", use_container_width=True):
        st.session_state.question = full

typed = st.chat_input("Or describe what happened in your own words")
if typed:
    st.session_state.question = typed

q = st.session_state.get("question")
if q:
    st.markdown(f'<p class="cs-asked">You asked: <b>{html.escape(q)}</b></p>', unsafe_allow_html=True)
    if compare:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<p class="cs-colhead">CyberShield SME <span>with ACSC sources</span></p>',
                        unsafe_allow_html=True)
            with st.spinner("Checking the ACSC guides..."):
                render_answer(load_bot(variant).answer(q))
        with right:
            st.markdown('<p class="cs-colhead">General AI <span>no sources</span></p>',
                        unsafe_allow_html=True)
            with st.spinner("Asking a general AI..."):
                render_answer(load_bot("llm_only").answer(q), grounded=False)
    else:
        with st.spinner("Checking the ACSC guides..."):
            render_answer(load_bot(variant).answer(q))
else:
    st.markdown('<p class="cs-foot">Pick the situation closest to yours, or type what happened. '
                'Every step links back to the ACSC guide it came from.</p>', unsafe_allow_html=True)
