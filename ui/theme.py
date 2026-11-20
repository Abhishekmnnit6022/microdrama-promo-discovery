"""Editorial light theme: one ink colour, one accent, generous whitespace."""

import streamlit as st

INK = "#16181D"
MUTED = "#5F6673"
RULE = "#E4E2DD"
PAPER = "#FBFAF7"
CARD = "#FFFFFF"
ACCENT = "#9A3412"      # burnt sienna, used sparingly
OK = "#2F6B4F"
WARN = "#9A6B12"
BAD = "#9B2C2C"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, td, th, label {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
h1, h2, h3 {{ font-family: 'Source Serif 4', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-size: 2.0rem !important; font-weight: 600 !important; }}
h2 {{ font-size: 1.35rem !important; font-weight: 600 !important; }}
h3 {{ font-size: 1.1rem !important; font-weight: 600 !important; }}
section[data-testid="stSidebar"] {{ background: #F3F1EC; border-right: 1px solid {RULE}; }}
.block-container {{ padding-top: 4.5rem !important; max-width: 1180px; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; line-height: 1.6; color: {MUTED}; font-weight: 600; padding-top: 2px; }}
.lede {{ color: {MUTED}; font-size: .98rem; max-width: 760px; margin-bottom: 1.2rem; }}
.kpi {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 6px; padding: 14px 16px; }}
.kpi .v {{ font-family: 'Source Serif 4', serif; font-size: 1.9rem; font-weight: 600; color: {INK}; line-height: 1.1; }}
.kpi .l {{ color: {MUTED}; font-size: .78rem; margin-top: 4px; }}
.card {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 8px; padding: 18px 20px; margin-bottom: 14px; }}
.card .head {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
.rank {{ font-family: 'Source Serif 4', serif; font-size: 1.6rem; font-weight: 600; color: {ACCENT}; }}
.score {{ font-size: .85rem; color: {INK}; font-weight: 600; }}
.meta {{ color: {MUTED}; font-size: .82rem; }}
.pill {{ display: inline-block; border: 1px solid {RULE}; border-radius: 999px; padding: 1px 9px; font-size: .72rem; color: {MUTED}; margin-right: 4px; }}
.pill.ok {{ color: {OK}; border-color: #BFD8CB; }} .pill.warn {{ color: {WARN}; border-color: #E8D6AE; }} .pill.bad {{ color: {BAD}; border-color: #E5C3C3; }}
.row2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 12px; }}
@media (max-width: 800px) {{ .row2 {{ grid-template-columns: 1fr; }} }}
.lbl {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: {MUTED}; font-weight: 600; margin-bottom: 3px; }}
.quote {{ font-family: 'Source Serif 4', serif; font-size: 1.0rem; color: {INK}; }}
.why {{ border-top: 1px solid {RULE}; margin-top: 12px; padding-top: 10px; font-size: .9rem; color: {INK}; }}
.flag {{ color: {WARN}; font-size: .82rem; }}
.bar {{ height: 6px; background: #EFEDE8; border-radius: 3px; overflow: hidden; }}
.bar > span {{ display: block; height: 6px; background: {INK}; }}
.brk {{ display: grid; grid-template-columns: 120px 1fr 40px; gap: 8px; align-items: center; font-size: .78rem; color: {MUTED}; margin: 2px 0; }}
hr {{ border-color: {RULE}; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 6px; border: 1px solid {INK}; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, lede: str = ""):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if lede:
        st.markdown(f'<div class="lede">{lede}</div>', unsafe_allow_html=True)
