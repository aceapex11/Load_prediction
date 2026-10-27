"""Look & feel: CSS, KPI cards, shared Plotly styling."""
import streamlit as st

AMBER, CYAN, VIOLET, GREEN, PINK, RED, BLUE, GREY = "#FFB547", "#4CC9F0", "#A78BFA", "#34D399", "#F472B6", "#FB7185", "#60A5FA", "#8892A6"
PHASE = {"RKW": RED, "YKW": "#FBBF24", "BKW": BLUE}
COLORWAY = [AMBER, CYAN, VIOLET, GREEN, PINK, BLUE, RED, GREY]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Space Grotesk', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0E1420; border-right: 1px solid #1B2335; }
.brand { display:flex; align-items:center; gap:.6rem; margin: .2rem 0 1.2rem 0; }
.brand .dot { width:34px; height:34px; border-radius:10px; background: linear-gradient(135deg,#FFB547,#FB7185); display:grid; place-items:center; color:#0B0F17; font-weight:700; }
.brand b { font-size:1.15rem; letter-spacing:.2px; } .brand span { display:block; font-size:.72rem; color:#8892A6; margin-top:-2px; }
.hero { padding: 1.4rem 1.6rem; border-radius: 18px; border:1px solid #1E2740;
  background: radial-gradient(900px 260px at 90% -20%, rgba(255,181,71,.16), transparent 60%), radial-gradient(700px 240px at 0% 120%, rgba(76,201,240,.12), transparent 60%), #111827; margin-bottom: 1.1rem; }
.hero h1 { margin:0; font-size:1.9rem; font-weight:700; letter-spacing:-.4px; }
.hero p { margin:.35rem 0 0; color:#9AA5BA; font-size:.98rem; max-width: 70ch; }
.chip { display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.74rem; border:1px solid #26304A; color:#B7C0D3; margin-right:.4rem; background:#0F1523; }
.kpi { background:#111827; border:1px solid #1E2740; border-radius:14px; padding:.9rem 1.05rem; height:100%; }
.kpi .l { color:#8892A6; font-size:.8rem; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:1.7rem; font-weight:600; margin-top:.15rem; letter-spacing:-.5px; }
.kpi .s { font-size:.78rem; margin-top:.1rem; color:#8892A6; }
.kpi.good .v { color:#34D399; } .kpi.warn .v { color:#FFB547; } .kpi.bad .v { color:#FB7185; } .kpi.info .v { color:#4CC9F0; }
.note { border-left:3px solid #FFB547; background:#141B2B; padding:.7rem 1rem; border-radius:0 10px 10px 0; color:#B7C0D3; font-size:.9rem; margin:.6rem 0 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid #1E2740; }
div[data-testid="stDataFrame"] { border:1px solid #1E2740; border-radius:12px; overflow:hidden; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.markdown('<div class="brand"><div class="dot">⚡</div><div><b>GridSense</b><span>Load forecasting &amp; balance</span></div></div>', unsafe_allow_html=True)


def hero(title, sub, chips=()):
    c = "".join(f'<span class="chip">{x}</span>' for x in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p><div style="margin-top:.8rem">{c}</div></div>', unsafe_allow_html=True)


def kpi(label, value, sub="", tone=""):
    st.markdown(f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def style(fig, height=420, legend=True):
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", size=13, color="#C9D1E3"), colorway=COLORWAY,
        margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"), showlegend=legend,
        xaxis=dict(gridcolor="#1B2335", zeroline=False), yaxis=dict(gridcolor="#1B2335", zeroline=False))
    return fig
