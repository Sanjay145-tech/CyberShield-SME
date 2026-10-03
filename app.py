"""CyberShield SME demo app.   Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag import CyberShield  # noqa: E402

st.set_page_config(page_title="CyberShield SME", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
:root { --ink:#1d2a3a; --muted:#5b6b7d; --steel:#2f5d7c; --amber:#b86e00; --paper:#f7f9fb; }
.block-container { max-width: 1100px; padding-top: 2rem; }
h1 { color: var(--ink); font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0; }
.tagline { color: var(--muted); font-size: 1.05rem; margin: .25rem 0 1.25rem; max-width: 62ch; }
.incident { display:inline-block; border-left: 4px solid var(--amber); padding: .15rem .6rem;
            background: #fff6e8; color: var(--ink); font-size: .9rem; margin-bottom: .5rem; }
.src { border-left: 3px solid var(--steel); padding-left: .6rem; margin: .4rem 0;
       color: var(--ink); font-size: .9rem; }
.src small { color: var(--muted); }
.note { color: var(--muted); font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

LABELS = {"phishing": "Phishing", "ransomware": "Ransomware",
          "bec": "Business email compromise", "account_compromise": "Account compromise"}

EXAMPLES = [
    "A supplier emailed saying their bank details changed. Should I pay the next invoice to the new account?",
    "An employee opened a suspicious attachment. What do we do right now?",
    "I got an MFA approval request I didn't start. Is my account compromised?",
    "Our files are encrypted and there's a ransom note. What should we do?",
]


@st.cache_resource(show_spinner="Loading the ACSC knowledge base...")
def load_bot(variant):
    return CyberShield(variant)


def render_answer(out, show_sources=True):
    if out["incident_types"]:
        names = ", ".join(LABELS.get(t, t) for t in out["incident_types"])
        st.markdown(f'<div class="incident">Looks like: {names}</div>', unsafe_allow_html=True)
    st.markdown(out["answer"])
    if show_sources and out["hits"]:
        cited = set(out["cited"])
        with st.expander(f"Sources ({len(cited)} cited)"):
            for i, h in enumerate(out["hits"], 1):
                mark = "Cited" if i in cited else "Retrieved, not cited"
                link = f' (<a href="{h["url"]}">open</a>)' if h.get("url") else ""
                st.markdown(
                    f'<div class="src"><b>[S{i}] {h["title"]}</b>, page {h["page"]}{link}<br>'
                    f'<small>{mark}</small><br>{h["text"][:400]}...</div>',
                    unsafe_allow_html=True)


st.markdown("# CyberShield SME")
st.markdown('<p class="tagline">Step-by-step help when your small business is hit by a scam '
            'or cyber attack, based only on official Australian Cyber Security Centre guidance.</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Demo settings**")
    compare = st.toggle("Compare with plain AI (no sources)", value=False)
    variant = st.selectbox("Pipeline", ["hybrid_scenario", "hybrid", "bm25_scenario", "bm25", "dense"],
                           help="hybrid_scenario is the full CyberShield pipeline.")
    st.markdown('<p class="note">In an emergency, contact the ACSC through cyber.gov.au. '
                'This prototype is a student project and not a replacement for professional '
                'incident response.</p>', unsafe_allow_html=True)

st.markdown("**Try a common situation**")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.question = ex

question = st.chat_input("Describe what happened...")
if question:
    st.session_state.question = question

q = st.session_state.get("question")
if q:
    st.markdown(f"**You asked:** {q}")
    if compare:
        left, right = st.columns(2)
        with left:
            st.markdown("#### CyberShield SME")
            with st.spinner("Checking ACSC guidance..."):
                render_answer(load_bot(variant).answer(q))
        with right:
            st.markdown("#### Plain AI, no sources")
            with st.spinner("Asking the model without sources..."):
                render_answer(load_bot("llm_only").answer(q), show_sources=False)
    else:
        with st.spinner("Checking ACSC guidance..."):
            render_answer(load_bot(variant).answer(q))
