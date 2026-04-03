
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles    import getSampleStyleSheet
from reportlab.platypus      import (SimpleDocTemplate, Paragraph,
                                      Spacer, Table, TableStyle)
from reportlab.lib           import colors

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CV_R2_NORM   = 0.355
TEST_R2_NORM = 0.361
N_SAMPLES    = 46_686
N_LOCATIONS  = 38
N_HYBRIDS    = 2_912
N_YEARS      = 5
MODEL_NAME   = "XGBoost"
DATASET      = "G2F 2014-2018"

st.set_page_config(
    page_title="NeuroCrop - Maize Yield Predictor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@300;600;800&family=Instrument+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Fraunces', serif; }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #071a0e 0%, #0d3320 40%, #14532d 75%, #166534 100%);
    border-radius: 20px; padding: 40px 48px 36px; margin-bottom: 28px;
    position: relative; overflow: hidden;
  }
  .hero::before {
    content: ""; position: absolute; top: 0; right: 0; bottom: 0; width: 45%;
    background: radial-gradient(ellipse at 80% 50%, rgba(74,222,128,0.12) 0%, transparent 70%);
  }
  .hero h1  { color: #f0fdf4; font-size: 2.6rem; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.02em; }
  .hero .sub { color: #a7f3c0; font-size: 0.95rem; font-weight: 400; letter-spacing: 0.04em; }
  .hero .tag {
    display: inline-block;
    background: rgba(74,222,128,0.18); border: 1px solid rgba(74,222,128,0.45);
    color: #6ee7a0; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    padding: 4px 12px; border-radius: 20px; margin-right: 6px; margin-top: 12px;
    text-transform: uppercase; cursor: default; transition: all 0.25s ease;
  }
  .hero .tag:hover {
    background: rgba(74,222,128,0.32); border-color: #4ade80;
    transform: translateY(-2px); box-shadow: 0 4px 14px rgba(74,222,128,0.25);
    color: #bbf7d0;
  }

  /* ── KPI cards ── */
  .kpi {
    background: #0d2218;
    border: 1.5px solid #1a4d2e;
    border-radius: 14px; padding: 18px 16px; text-align: center;
    transition: all 0.25s ease; cursor: default;
  }
  .kpi:hover {
    background: #112b1e;
    border-color: #22c55e;
    box-shadow: 0 6px 22px rgba(34,197,94,0.2);
    transform: translateY(-3px);
  }
  .kpi .v { font-size: 1.9rem; font-weight: 700; color: #4ade80; font-family: 'Fraunces', serif; }
  .kpi .l { font-size: 0.72rem; color: #86efac; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; font-weight: 500; }

  /* ── Info / Warn pills ── */
  .info-pill {
    background: #0a1f12;
    border-left: 4px solid #22c55e;
    border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 10px 0;
    font-size: 0.89rem; color: #d1fae5; line-height: 1.7; font-weight: 400;
    transition: border-left-width 0.2s, padding-left 0.2s, background 0.2s;
  }
  .info-pill strong { color: #4ade80; }
  .info-pill:hover { border-left-width: 6px; padding-left: 22px; background: #0d2218; }

  .warn-pill {
    background: #1a1200;
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 10px 0;
    font-size: 0.89rem; color: #fde68a; line-height: 1.7; font-weight: 400;
    transition: border-left-width 0.2s, padding-left 0.2s, background 0.2s;
  }
  .warn-pill:hover { border-left-width: 6px; padding-left: 22px; background: #221800; }

  /* ── Compare cards ── */
  .compare-card {
    background: #0d2218;
    border: 1.5px solid #1a4d2e;
    border-radius: 14px; padding: 20px; text-align: center;
    transition: all 0.25s ease;
  }
  .compare-card:hover {
    background: #112b1e;
    border-color: #22c55e;
    box-shadow: 0 6px 24px rgba(34,197,94,0.2);
    transform: translateY(-3px);
  }
  .compare-card .hn { font-family: 'Fraunces', serif; font-size: 1.1rem; color: #a7f3c0; font-weight: 700; }
  .compare-card .yb { font-size: 2.4rem; font-weight: 700; font-family: 'Fraunces', serif; color: #4ade80; }
  .compare-card .yu { font-size: 0.85rem; color: #86efac; font-weight: 500; }

  /* ── About sections ── */
  .about-section {
    background: #0d2218;
    border: 1.5px solid #1a4d2e;
    border-radius: 14px; padding: 24px 28px; margin: 12px 0;
    transition: all 0.25s ease;
  }
  .about-section:hover {
    background: #112b1e;
    border-color: #22c55e;
    box-shadow: 0 4px 20px rgba(34,197,94,0.15);
  }
  .about-section h4 { font-family: 'Fraunces', serif; color: #4ade80; margin-bottom: 10px; font-size: 1.05rem; font-weight: 700; }
  .about-section, .about-section p { color: #d1fae5; line-height: 1.75; }
  .about-section strong { color: #a7f3c0; }
  .about-section a { color: #4ade80; font-weight: 600; text-decoration: none; }
  .about-section a:hover { text-decoration: underline; color: #86efac; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f0fdf4; border-radius: 10px; padding: 5px;
    border: 1.5px solid #bbf7d0;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 7px 16px;
    font-weight: 500; font-size: 0.85rem;
    color: #166534 !important;
    transition: all 0.2s;
  }
  .stTabs [data-baseweb="tab"]:hover { background: #dcfce7 !important; }
  .stTabs [aria-selected="true"] {
    background: #16a34a !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(22,163,74,0.3);
  }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for name in ["all_predictions.csv.gz", "all_predictions.csv"]:
        p = ROOT / "outputs" / "predictions" / name
        if p.exists():
            return pd.read_csv(p)
    st.error("**all_predictions.csv.gz not found.**\n\nRun Cell 37 in Colab, then push `outputs/predictions/all_predictions.csv.gz` to GitHub.")
    st.stop()

df        = load_data()
females   = sorted(df["Female"].unique().tolist())
males     = sorted(df["Male"].unique().tolist())
locations = sorted(df["Location"].unique().tolist())

@st.cache_data
def ov():
    return {"mean": df["Yield"].mean(), "std": df["Yield"].std(),
            "min": df["Yield"].min(), "max": df["Yield"].max(), "n": len(df)}

OV = ov()

def pct_rank(y, loc):
    sub = df[df["Location"]==loc]["Yield"]
    return round(float((sub < y).mean())*100, 1) if len(sub) else 0.

def cat(y):
    if y >= 170: return "🟢 High"
    if y >= 150: return "🟡 Medium"
    return "🔴 Low"

def lookup(p1, p2, loc):
    r = df[(df["Female"]==p1)&(df["Male"]==p2)&(df["Location"]==loc)]
    if not len(r):
        r = df[(df["Female"]==p2)&(df["Male"]==p1)&(df["Location"]==loc)]
    return round(float(r.iloc[0]["Yield"]), 2) if len(r) else None

@st.cache_data
def stability_df():
    g = df.groupby(["Female","Male"])["Yield"]
    t = g.agg(Mean_Yield="mean", Std_Yield="std", N_Locs="count").reset_index()
    t["CV_pct"]    = (t["Std_Yield"]/t["Mean_Yield"]*100).round(1)
    t["Hybrid"]    = t["Female"]+" x "+t["Male"]
    t["Stability"] = t["CV_pct"].apply(lambda v: "🟢 Stable" if v<5 else ("🟡 Moderate" if v<10 else "🔴 Unstable"))
    return t.sort_values("Mean_Yield", ascending=False).reset_index(drop=True)

# ── PDF ───────────────────────────────────────────────────────
def make_pdf(p1, p2, loc, pred, loc_rows, percentile):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    sty = getSampleStyleSheet(); s = []
    s.append(Paragraph("NeuroCrop - Maize Yield Prediction Report", sty["Title"]))
    s.append(Spacer(1,10))
    s.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Model: {MODEL_NAME}  |  Dataset: {DATASET}", sty["Normal"]))
    s.append(Spacer(1,14))
    s.append(Paragraph("Model Performance", sty["Heading2"]))
    s.append(Paragraph(f"CV R2 (honest) = {CV_R2_NORM:.3f}  |  Test R2 = {TEST_R2_NORM:.3f}  |  Samples = {N_SAMPLES:,}  |  Locations = {N_LOCATIONS}  |  Years = {N_YEARS}", sty["Normal"]))
    s.append(Spacer(1,14))
    s.append(Paragraph("Prediction Summary", sty["Heading2"]))
    cs = cat(pred).replace("🟢","").replace("🟡","").replace("🔴","").strip()
    data = [["Parameter","Value"],["Female Parent",p1],["Male Parent",p2],["Location",loc],
            ["Predicted Yield",f"{pred} bu/A"],["Percentile",f"Top {100-percentile:.0f}%"],["Category",cs]]
    tbl = Table(data, colWidths=[200,300])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#14532d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),1,colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f0fdf4")]),
    ]))
    s.append(tbl); s.append(Spacer(1,20))
    if loc_rows:
        s.append(Paragraph("Top Locations", sty["Heading2"]))
        ld = [["Rank","Location","Yield (bu/A)","Category"]]
        for i,r in enumerate(loc_rows[:10],1):
            ld.append([str(i),r["Location"],str(r["Yield"]),cat(r["Yield"]).replace("🟢","").replace("🟡","").replace("🔴","").strip()])
        lt = Table(ld, colWidths=[50,150,150,150])
        lt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#14532d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                  ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),1,colors.grey),
                                  ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f0fdf4")])]))
        s.append(lt)
    s.append(Spacer(1,30))
    s.append(Paragraph(f"NeuroCrop - Generative Breeding Platform  |  Abdul Manan  |  {DATASET}", sty["Normal"]))
    doc.build(s); buf.seek(0); return buf

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>🌽 NeuroCrop</h1>
  <div class="sub">GENERATIVE BREEDING PLATFORM  ·  MAIZE HYBRID YIELD PREDICTION</div>
  <div>
    <span class="tag">XGBoost</span><span class="tag">Genomics + Environment</span>
    <span class="tag">5-Year Multi-Location</span><span class="tag">G×E Modelling</span>
  </div>
</div>
""", unsafe_allow_html=True)

cols = st.columns(7)
for col, val, lbl in [
    (cols[0], f"{CV_R2_NORM:.3f}", "CV R² (honest)"),
    (cols[1], f"{TEST_R2_NORM:.3f}", "Test R²"),
    (cols[2], f"{N_SAMPLES:,}", "Samples"),
    (cols[3], f"{N_LOCATIONS}", "Locations"),
    (cols[4], f"{N_HYBRIDS:,}", "Hybrids"),
    (cols[5], f"{N_YEARS} yrs", "G2F Years"),
    (cols[6], f"{OV['n']:,}", "Predictions"),
]:
    col.markdown(f'<div class="kpi"><div class="v">{val}</div><div class="l">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 🔬 Select Hybrid")
default_f = females.index("B73")  if "B73"  in females else 0
default_m = males.index("Mo17")   if "Mo17" in males   else 0
female   = st.sidebar.selectbox("Female Parent", females, index=default_f)
male     = st.sidebar.selectbox("Male Parent",   males,   index=default_m)
location = st.sidebar.selectbox("Location",      locations)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** {MODEL_NAME}  \n**CV R²:** {CV_R2_NORM:.3f}  \n**Test R²:** {TEST_R2_NORM:.3f}  \n**Samples:** {N_SAMPLES:,}  \n**Hybrids:** {N_HYBRIDS:,}")
st.sidebar.markdown("---")
st.sidebar.caption("NeuroCrop · Abdul Manan")

# ══════════════════════════════════════════════════════════════
# TABS — 10 total (6 existing + 4 new)
# ══════════════════════════════════════════════════════════════
(tab1, tab2, tab3, tab4, tab5,
 tab6, tab7, tab8, tab9, tab10) = st.tabs([
    "🔮 Predict",
    "📍 Best Location",
    "🏆 Best Cross",
    "⚖️ Compare Hybrids",    # NEW
    "🔄 G×E Analysis",
    "📊 Stability",
    "📈 Yield Explorer",     # NEW
    "📦 Batch Predict",
    "🧠 Model Insights",     # NEW
    "ℹ️ About",               # NEW
])

# TAB 1 — Predict ─────────────────────────────────────────────
with tab1:
    st.subheader(f"Prediction: {female} x {male} @ {location}")
    pred = lookup(female, male, location)
    if pred:
        p = pct_rank(pred, location)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Predicted Yield",  f"{pred} bu/A")
        m2.metric("Percentile Rank",  f"Top {100-p:.0f}%", help="vs all crosses at this location")
        m3.metric("vs Overall Avg",   f"{pred-OV['mean']:+.1f} bu/A")
        m4.metric("Category",         cat(pred))
        c = cat(pred)
        if "High" in c:   st.success(f"{c} Yield")
        elif "Medium" in c: st.warning(f"{c} Yield")
        else:               st.error(f"{c} Yield")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred,
            title={"text":"Predicted Yield (bu/A)","font":{"size":15}},
            delta={"reference":OV["mean"],"suffix":" vs avg"},
            gauge={"axis":{"range":[OV["min"],OV["max"]]},"bar":{"color":"#16a34a"},
                   "steps":[{"range":[OV["min"],150],"color":"#fde8e8"},{"range":[150,170],"color":"#fef9c3"},{"range":[170,OV["max"]],"color":"#dcfce7"}],
                   "threshold":{"line":{"color":"#14532d","width":3},"thickness":0.8,"value":OV["mean"]}}))
        fig.update_layout(
            height=330, margin=dict(t=40,b=10),
            paper_bgcolor="#0d1f13", plot_bgcolor="#0d1f13",
            font=dict(color="#e2f5e9", size=13)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="info-pill"><strong>Feature contribution:</strong> Genetics 41% · Plant traits 24% · Season weather 19% · Critical-period weather 16%</div>', unsafe_allow_html=True)
        st.markdown("---")
        @st.cache_data
        def _pdf_locs(p1, p2):
            return sorted([{"Location":l,"Yield":v} for l in locations if (v:=lookup(p1,p2,l))], key=lambda x: x["Yield"], reverse=True)
        pdf = make_pdf(female, male, location, pred, _pdf_locs(female, male), p)
        st.download_button("📄 Download PDF Report", pdf, f"neurocrop_{female}_{male}_{location}.pdf", "application/pdf", use_container_width=True)
    else:
        st.warning("Combination not found in database.")

# TAB 2 — Best Location ───────────────────────────────────────
with tab2:
    st.subheader(f"Best Locations for {female} x {male}")
    @st.cache_data
    def _best_locs(p1, p2):
        rows = [{"Location":l,"Yield":v,"Percentile":pct_rank(v,l),"Category":cat(v)} for l in locations if (v:=lookup(p1,p2,l))]
        return pd.DataFrame(rows).sort_values("Yield", ascending=False).reset_index(drop=True)
    res = _best_locs(female, male)
    if len(res):
        res.index += 1
        b = res.iloc[0]
        st.success(f"🏆 Best: **{b['Location']}** → {b['Yield']} bu/A  (Top {100-b['Percentile']:.0f}%)")
        fig = px.bar(res, x="Location", y="Yield", color="Yield", color_continuous_scale="RdYlGn",
                     title=f"{female} x {male} — Yield by Location", text="Yield", template="plotly_dark")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                          textfont=dict(size=11, color="#e2f5e9"))
        fig.update_layout(
            xaxis_tickangle=-45, height=520, showlegend=False,
            plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13",
            font=dict(color="#e2f5e9", size=12),
            title=dict(font=dict(size=16, color="#a7f3c0")),
            xaxis=dict(tickfont=dict(size=11, color="#e2f5e9"), gridcolor="#1e3a28"),
            yaxis=dict(tickfont=dict(size=12, color="#e2f5e9"), gridcolor="#1e3a28",
                       title="Yield (bu/A)", title_font=dict(color="#86efac")),
            margin=dict(t=60, b=100)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res, use_container_width=True)
        st.download_button("📥 Download CSV", res.to_csv(index=False), f"best_locs_{female}_{male}.csv")

# TAB 3 — Best Cross ──────────────────────────────────────────
with tab3:
    st.subheader(f"Top Crosses at {location}")
    top_n = st.slider("Show top N", 5, 50, 20)
    cross_df = df[df["Location"]==location].copy()
    cross_df["Cross"] = cross_df["Female"]+" x "+cross_df["Male"]
    cross_df = cross_df.sort_values("Yield", ascending=False).head(top_n).reset_index(drop=True)
    cross_df.index += 1
    fig = px.bar(cross_df, x="Yield", y="Cross", orientation="h", color="Yield",
                 color_continuous_scale="RdYlGn", title=f"Top {top_n} Crosses at {location}",
                 text="Yield", template="plotly_dark")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(height=max(400,top_n*25), showlegend=False, yaxis={"categoryorder":"total ascending"}, plot_bgcolor="#0d1f13")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cross_df[["Cross","Yield"]], use_container_width=True)
    st.download_button("📥 Download CSV", cross_df.to_csv(index=False), f"top_crosses_{location}.csv")

# TAB 4 — Compare Hybrids (NEW) ───────────────────────────────
with tab4:
    st.subheader("⚖️ Side-by-Side Hybrid Comparison")
    st.markdown('<div class="info-pill">Compare up to 4 hybrids — yields, location rankings, and stability metrics. Great for choosing between breeding candidates before committing to field trials.</div>', unsafe_allow_html=True)
    n_comp = st.radio("Number of hybrids to compare", [2, 3, 4], horizontal=True)
    hybs   = []
    cols   = st.columns(n_comp)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Hybrid {i+1}**")
            fi = col.selectbox(f"Female {i+1}", females, key=f"cf{i}", index=min(i*3,len(females)-1))
            mi = col.selectbox(f"Male {i+1}",   males,   key=f"cm{i}", index=min(i*2,len(males)-1))
            li = col.selectbox(f"Location {i+1}", locations, key=f"cl{i}")
            hybs.append((fi, mi, li))
    st.markdown("---")
    card_cols = st.columns(n_comp)
    comp_rows = []
    for i,(fi,mi,li) in enumerate(hybs):
        v = lookup(fi, mi, li)
        with card_cols[i]:
            if v:
                p_ = pct_rank(v, li); c_ = cat(v)
                color = "#16a34a" if "High" in c_ else ("#f59e0b" if "Medium" in c_ else "#dc2626")
                st.markdown(f'<div class="compare-card"><div class="hn">{fi} x {mi}</div><div style="font-size:0.75rem;color:#6b7280;margin-bottom:8px">{li}</div><div class="yb" style="color:{color}">{v}</div><div class="yu">bu/A</div><div style="margin-top:8px;font-size:0.8rem;color:#4b7a5e">Top {100-p_:.0f}% at location</div><div style="font-size:0.85rem;margin-top:4px">{c_}</div></div>', unsafe_allow_html=True)
                comp_rows.append({"Hybrid":f"{fi} x {mi}","Location":li,"Yield":v,"Percentile":100-p_,"Category":c_})
            else:
                st.warning(f"No data for {fi} x {mi} @ {li}")
    if len(comp_rows) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        comp_df = pd.DataFrame(comp_rows)
        comp_df["Label"] = comp_df["Hybrid"]+"\n@"+comp_df["Location"]
        fig = px.bar(comp_df, x="Label", y="Yield", color="Yield", color_continuous_scale="RdYlGn",
                     text="Yield", title="Yield Comparison", template="plotly_dark")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(height=360, showlegend=False, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13), xaxis_title="", yaxis_title="Predicted Yield (bu/A)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Performance across ALL locations:**")
        prof = []
        for fi,mi,_ in hybs:
            for l in locations:
                v = lookup(fi,mi,l)
                if v: prof.append({"Hybrid":f"{fi} x {mi}","Location":l,"Yield":v})
        if prof:
            prof_df = pd.DataFrame(prof)
            fig2 = px.line(prof_df, x="Location", y="Yield", color="Hybrid", markers=True,
                           title="Yield Profile Across All Locations", template="plotly_dark")
            fig2.update_layout(height=400, xaxis_tickangle=-45, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13))
            st.plotly_chart(fig2, use_container_width=True)
        stab = stability_df()
        summ = []
        for fi,mi,li in hybs:
            v = lookup(fi,mi,li); row = stab[(stab["Female"]==fi)&(stab["Male"]==mi)]
            summ.append({"Hybrid":f"{fi} x {mi}","Location":li,"Predicted":v or "—",
                         "Mean (all locs)":f"{row['Mean_Yield'].values[0]:.1f}" if len(row) else "—",
                         "CV%":f"{row['CV_pct'].values[0]:.1f}" if len(row) else "—",
                         "Stability":row["Stability"].values[0] if len(row) else "—"})
        st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", comp_df.to_csv(index=False), "comparison.csv")

# TAB 5 — G×E Analysis ────────────────────────────────────────
with tab5:
    st.subheader("G×E Interaction Analysis")
    st.markdown('<div class="info-pill">Crossing lines = strong G×E (location-specific adaptation). Parallel lines = stable, wide-adapted hybrid.</div>', unsafe_allow_html=True)
    sel_f = st.multiselect("Select Female Parents", females, default=females[:3])
    fix_m = st.selectbox("Fixed Male Parent", males, index=males.index("Mo17") if "Mo17" in males else 0, key="ge_male")
    if sel_f:
        ge = df[df["Female"].isin(sel_f)&(df["Male"]==fix_m)].copy()
        ge["Hybrid"] = ge["Female"]+" x "+ge["Male"]
        if len(ge):
            fig = px.line(ge, x="Location", y="Yield", color="Hybrid", markers=True, title="G×E Interaction", template="plotly_dark")
            fig.add_hline(y=ge["Yield"].mean(), line_dash="dash", line_color="#14532d", annotation_text=f"Grand Mean ({ge['Yield'].mean():.1f} bu/A)")
            fig.update_layout(height=480, xaxis_tickangle=-45, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13))
            st.plotly_chart(fig, use_container_width=True)
            pivot = ge.pivot_table(index="Hybrid", columns="Location", values="Yield", aggfunc="mean")
            if not pivot.empty:
                fig2 = px.imshow(pivot, color_continuous_scale="RdYlGn", title="G×E Heatmap (bu/A)", text_auto=".0f", template="plotly_dark")
                fig2.update_layout(
                    height=max(320, len(sel_f)*80+140),
                    paper_bgcolor="#0d1f13", plot_bgcolor="#0d1f13",
                    font=dict(color="#e2f5e9", size=13),
                    title=dict(font=dict(size=15, color="#a7f3c0"))
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for the selected combination.")

# TAB 6 — Stability ───────────────────────────────────────────
with tab6:
    st.subheader("G×E Stability Ranking")
    st.markdown('<div class="info-pill"><strong>CV%</strong> = coefficient of variation across locations. Lower = stable everywhere. Core of generative breeding: high-yielding AND stable crosses reduce trial costs.</div>', unsafe_allow_html=True)
    stab = stability_df()
    c1, c2 = st.columns(2)
    min_m = c1.slider("Min mean yield (bu/A)", int(df["Yield"].min()), int(df["Yield"].max()), 150)
    max_c = c2.slider("Max CV%", 1, 30, 10)
    filt  = stab[(stab["Mean_Yield"]>=min_m)&(stab["CV_pct"]<=max_c)].head(50)
    if len(filt):
        st.success(f"**{len(filt)} hybrids** — high yield + stable")
        fig = px.scatter(filt, x="CV_pct", y="Mean_Yield", color="Stability",
                         hover_data=["Hybrid","N_Locs"],
                         color_discrete_map={"🟢 Stable":"#16a34a","🟡 Moderate":"#ca8a04","🔴 Unstable":"#dc2626"},
                         title="Yield vs Stability", labels={"CV_pct":"CV% (lower = more stable)","Mean_Yield":"Mean Yield (bu/A)"},
                         template="plotly_dark")
        fig.update_layout(height=440, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13))
        st.plotly_chart(fig, use_container_width=True)
        show = ["Hybrid","Mean_Yield","Std_Yield","CV_pct","N_Locs","Stability"]
        st.dataframe(filt[show].reset_index(drop=True), use_container_width=True)
        st.download_button("📥 Download Table", filt[show].to_csv(index=False), "stability.csv")
    else:
        st.warning("No hybrids match. Try relaxing the filters.")

# TAB 7 — Yield Explorer (NEW) ────────────────────────────────
with tab7:
    st.subheader("📈 Yield Explorer — Database Analytics")
    st.markdown('<div class="info-pill">Explore yield distributions, location rankings, and parent effects across all predictions.</div>', unsafe_allow_html=True)
    et1, et2, et3 = st.tabs(["Distribution", "By Location", "Parent Effects"])

    with et1:
        fig = px.histogram(df, x="Yield", nbins=60,
                           title=f"Global Yield Distribution ({OV['n']:,} predictions)",
                           color_discrete_sequence=["#16a34a"], template="plotly_dark")
        fig.add_vline(x=OV["mean"], line_dash="dash", line_color="#14532d", annotation_text=f"Mean: {OV['mean']:.1f}")
        fig.add_vline(x=150, line_dash="dot", line_color="orange",  annotation_text="Medium threshold")
        fig.add_vline(x=170, line_dash="dot", line_color="#16a34a", annotation_text="High threshold")
        fig.update_layout(height=380, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13))
        st.plotly_chart(fig, use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean",  f"{OV['mean']:.1f} bu/A")
        c2.metric("Std",   f"{OV['std']:.1f} bu/A")
        c3.metric("Min",   f"{OV['min']:.1f} bu/A")
        c4.metric("Max",   f"{OV['max']:.1f} bu/A")
        ph = (df["Yield"]>=170).mean()*100; pm = ((df["Yield"]>=150)&(df["Yield"]<170)).mean()*100; pl = (df["Yield"]<150).mean()*100
        fig2 = px.pie(values=[ph,pm,pl], names=["High (>=170)","Medium (150-170)","Low (<150)"],
                      color_discrete_sequence=["#16a34a","#ca8a04","#dc2626"], title="Category Distribution", hole=0.45, template="plotly_dark")
        fig2.update_layout(
            height=320, paper_bgcolor="#0d1f13",
            font=dict(color="#e2f5e9", size=13),
            title=dict(font=dict(size=15, color="#a7f3c0")),
            legend=dict(font=dict(size=12, color="#e2f5e9"))
        )
        st.plotly_chart(fig2, use_container_width=True)

    with et2:
        ls = df.groupby("Location")["Yield"].agg(Mean="mean", Std="std", N="count").reset_index().sort_values("Mean", ascending=False)
        fig = px.bar(ls, x="Location", y="Mean", error_y="Std", color="Mean", color_continuous_scale="RdYlGn",
                     title="Mean Predicted Yield by Location (+/-1 std)", template="plotly_dark")
        fig.update_layout(height=460, xaxis_tickangle=-45, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ls.reset_index(drop=True), use_container_width=True)
        st.download_button("📥 Location Stats CSV", ls.to_csv(index=False), "location_stats.csv")

    with et3:
        st.markdown("**Top 20 Female Parents by Mean Yield**")
        fs = df.groupby("Female")["Yield"].mean().sort_values(ascending=False).head(20).reset_index()
        fs.columns = ["Female","Mean Yield"]
        fig = px.bar(fs, x="Female", y="Mean Yield", color="Mean Yield", color_continuous_scale="Greens",
                     title="Top 20 Female Parents", template="plotly_dark")
        fig.update_layout(height=360, xaxis_tickangle=-45, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Top 20 Male Parents by Mean Yield**")
        ms = df.groupby("Male")["Yield"].mean().sort_values(ascending=False).head(20).reset_index()
        ms.columns = ["Male","Mean Yield"]
        fig2 = px.bar(ms, x="Male", y="Mean Yield", color="Mean Yield", color_continuous_scale="Blues",
                      title="Top 20 Male Parents", template="plotly_dark")
        fig2.update_layout(height=360, xaxis_tickangle=-45, plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# TAB 8 — Batch Predict ───────────────────────────────────────
with tab8:
    st.subheader("📦 Batch Yield Prediction")
    st.markdown("Upload a CSV with columns: **Female, Male, Location**")
    sample = pd.DataFrame({"Female":["B73","A632","Oh43"],"Male":["Mo17","Mo17","Mo17"],"Location":["ILH1","WIH1","IAH4"]})
    st.dataframe(sample, use_container_width=True)
    st.download_button("📥 Download Template", sample.to_csv(index=False), "template.csv", "text/csv")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        inp = pd.read_csv(uploaded)
        st.markdown(f"**{len(inp)} rows**")
        st.dataframe(inp.head(), use_container_width=True)
        miss = [c for c in ["Female","Male","Location"] if c not in inp.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        elif st.button("🔮 Run All Predictions", type="primary", use_container_width=True):
            results, errors = [], []
            prog = st.progress(0); stat = st.empty()
            for i, row in inp.iterrows():
                p1, p2, loc = str(row["Female"]).strip(), str(row["Male"]).strip(), str(row["Location"]).strip()
                v = lookup(p1, p2, loc)
                if v: results.append({"Female":p1,"Male":p2,"Location":loc,"Predicted Yield":v,"Percentile":pct_rank(v,loc),"Category":cat(v)})
                else: errors.append({"Female":p1,"Male":p2,"Location":loc,"Error":"Not found"})
                prog.progress((i+1)/len(inp)); stat.text(f"Processing {i+1}/{len(inp)}...")
            prog.empty(); stat.empty()
            if results:
                res = pd.DataFrame(results).sort_values("Predicted Yield", ascending=False).reset_index(drop=True)
                res.index += 1
                st.success(f"✅ {len(results)} predictions")
                if errors: st.warning(f"⚠️ {len(errors)} not found")
                m1,m2,m3 = st.columns(3)
                m1.metric("Best",    f"{res['Predicted Yield'].max():.1f} bu/A")
                m2.metric("Average", f"{res['Predicted Yield'].mean():.1f} bu/A")
                m3.metric("Worst",   f"{res['Predicted Yield'].min():.1f} bu/A")
                st.dataframe(res, use_container_width=True)
                fig = px.histogram(res, x="Predicted Yield", nbins=20, title="Batch Distribution",
                                   color_discrete_sequence=["#16a34a"], template="plotly_dark")
                fig.update_layout(paper_bgcolor="#0d1f13", plot_bgcolor="#0d1f13", font=dict(color="#e2f5e9", size=13))
                st.plotly_chart(fig, use_container_width=True)
                c1,c2 = st.columns(2)
                c1.download_button("📥 CSV", res.to_csv(index=False), "batch.csv", "text/csv", use_container_width=True)
                xb = io.BytesIO()
                with pd.ExcelWriter(xb, engine="openpyxl") as w: res.to_excel(w, index=False, sheet_name="Predictions")
                xb.seek(0)
                c2.download_button("📊 Excel", xb, "batch.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                if errors:
                    with st.expander(f"❌ {len(errors)} failed"): st.dataframe(pd.DataFrame(errors), use_container_width=True)
            else:
                st.error("No predictions found. Check Female/Male/Location values match the database.")

# TAB 9 — Model Insights (NEW) ────────────────────────────────
with tab9:
    st.subheader("🧠 Model Insights")
    st.markdown('<div class="info-pill">Understand what the model learned and how NeuroCrop compares to standard GBLUP tools.</div>', unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown("#### Feature Importance")
        fi_df = pd.DataFrame({
            "Feature": ["Genetics (SNP PCA)", "Plant Traits", "Season Weather", "Critical-Period Weather", "Soil"],
            "Importance %": [41.1, 23.7, 18.9, 16.3, 0.0]
        })
        fig = px.bar(
            fi_df, x="Importance %", y="Feature", orientation="h",
            color="Importance %", color_continuous_scale="Greens",
            template="plotly_dark", title="XGBoost Feature Importance",
            text="Importance %"
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%", textposition="outside",
            textfont=dict(size=14, color="#e2f5e9"),
            marker_line_color="#4ade80", marker_line_width=1.2
        )
        fig.update_layout(
            height=380,
            showlegend=False,
            plot_bgcolor="#0d1f13",
            paper_bgcolor="#0d1f13",
            font=dict(color="#e2f5e9", size=13),
            title=dict(font=dict(size=16, color="#a7f3c0")),
            xaxis=dict(
                title="Importance %", title_font=dict(size=13, color="#86efac"),
                tickfont=dict(size=12, color="#e2f5e9"),
                range=[0, 52], gridcolor="#1e3a28", showgrid=True
            ),
            yaxis=dict(
                categoryorder="total ascending",
                tickfont=dict(size=13, color="#e2f5e9"),
                title=""
            ),
            margin=dict(l=180, r=60, t=50, b=30),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 2017 vs 5-Year Model")
        perf = pd.DataFrame({"Version":["2017 (RF)","2014-2018 (XGBoost)"],"CV R2":[0.572,0.355],"Test R2":[0.635,0.361]})
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="CV R2",   x=perf["Version"], y=perf["CV R2"],   marker_color="#16a34a",
                              text=[f"{v:.3f}" for v in perf["CV R2"]], textposition="outside",
                              textfont=dict(size=13, color="#e2f5e9")))
        fig2.add_trace(go.Bar(name="Test R2", x=perf["Version"], y=perf["Test R2"], marker_color="#4ade80",
                              text=[f"{v:.3f}" for v in perf["Test R2"]], textposition="outside",
                              textfont=dict(size=13, color="#e2f5e9")))
        fig2.update_layout(
            barmode="group", template="plotly_dark", height=340,
            plot_bgcolor="#0d1f13", paper_bgcolor="#0d1f13",
            font=dict(color="#e2f5e9", size=13),
            title=dict(text="Model Performance Comparison", font=dict(size=16, color="#a7f3c0")),
            yaxis=dict(title="R²", title_font=dict(size=13, color="#86efac"),
                       tickfont=dict(size=12, color="#e2f5e9"), range=[0, 0.8], gridcolor="#1e3a28"),
            xaxis=dict(tickfont=dict(size=13, color="#e2f5e9")),
            legend=dict(font=dict(size=12, color="#e2f5e9"), bgcolor="#0d1f13", bordercolor="#1e3a28"),
            margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        st.markdown("#### What Changed: 2017 → 2014–2018")
        chg = [("Samples","2,867","46,686","16x more"),("Years","1","5","Multi-year G×E"),
               ("Locations","23","38","+15 environments"),("Hybrids","654","2,912","4.5x diversity"),
               ("Algorithm","RF","XGBoost","Better G×E handling"),("SNP strategy","Concat","Mid-parent","Half RAM"),
               ("CV (honest)","0.572*","0.355","*had data leakage"),("Predictions","~100k","2,994,894","Full coverage")]
        st.dataframe(pd.DataFrame(chg, columns=["Metric","2017","2014-2018","Reason"]), use_container_width=True, hide_index=True)

        st.markdown("#### NeuroCrop vs Industry GBLUP")
        st.markdown('<div class="info-pill"><strong>Standard GBLUP</strong> (Pioneer, Bayer): pedigree + genomics. No environment data.<br><br><strong>NeuroCrop</strong>: adds actual field-level climate (daily temp, rainfall, solar radiation May-Sep + Jun-Aug) + soil. Environment contributes 35% of predictive power.</div>', unsafe_allow_html=True)
        st.markdown('<div class="warn-pill">CV R2 dropped from 0.572 (2017) to 0.355 (5-year) because the old CV had data leakage. Current 0.355 is the honest number. Published GBLUP benchmarks on G2F: R2 = 0.35-0.55.</div>', unsafe_allow_html=True)

# TAB 10 — About (NEW) ────────────────────────────────────────
with tab10:
    st.subheader("ℹ️ About NeuroCrop")
    ca, cb = st.columns([2,1])
    with ca:
        st.markdown("""
        <div class="about-section">
          <h4>What is NeuroCrop?</h4>
          A generative breeding platform predicting maize hybrid grain yield before field trials,
          using genomic SNPs, weather, soil, and plant traits. Trained on the public G2F dataset:
          5 years (2014-2018), 38 US locations, 2,912 hybrids, 2,994,894 pre-computed predictions.
        </div>
        <div class="about-section">
          <h4>Technical Architecture</h4>
          <strong>Genomics:</strong> Top 10k SNPs (variance filter from 437k-SNP VCF). Mid-parent average compressed to 20 PCA components via TruncatedSVD.<br><br>
          <strong>Environment:</strong> Season weather (May-Sep) + critical-period weather (Jun-Aug) + soil = 27 features.<br><br>
          <strong>Model:</strong> XGBoost (400 trees, lr=0.03). Per-location z-score normalisation. Honest 3-fold CV.<br><br>
          <strong>Inference:</strong> 2,994,894 pre-computed predictions stored in a 30 MB .csv.gz. No model inference at runtime.
        </div>
        <div class="about-section">
          <h4>Dataset</h4>
          Public G2F (Genomes to Fields) initiative. DOI: 10.25739/ragt-7213<br>
          VCF: inbreds_G2F_2014-2023_437k.vcf — 2,193 inbreds, 437,214 SNPs
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown("""
        <div class="about-section">
          <h4>Author</h4>
          <strong>Abdul Manan</strong><br>
          Plant Breeder · ML Researcher<br>
          Generative Breeding Startup<br><br>
          📧 abdulmanan2287@gmail.com<br>
          🔗 <a href="https://www.linkedin.com/in/abdul-manan-0aa546332/">LinkedIn</a><br>
          💻 <a href="https://github.com/manan348">GitHub</a>
        </div>
        <div class="about-section">
          <h4>Metrics</h4>
          CV R² = 0.355 (honest)<br>
          Test R² = 0.361<br>
          Samples = 46,686<br>
          Locations = 38<br>
          Hybrids = 2,912<br>
          Years = 5
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"**NeuroCrop** · Generative Breeding · {DATASET} · {MODEL_NAME} · "
    f"CV R² = {CV_R2_NORM:.3f} · Test R² = {TEST_R2_NORM:.3f} · "
    f"{N_SAMPLES:,} samples · {N_LOCATIONS} locations · "
    "**Abdul Manan** · [GitHub](https://github.com/manan348) · "
    "[LinkedIn](https://www.linkedin.com/in/abdul-manan-0aa546332/)"
)
