
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import datetime
from pathlib import Path

# ── PDF imports ────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles    import getSampleStyleSheet
from reportlab.platypus      import (SimpleDocTemplate, Paragraph,
                                      Spacer, Table, TableStyle)
from reportlab.lib           import colors

# ── Path resolution ────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# ── Constants — updated from v25 notebook (5-year multi-location run) ─────
CV_R2_NORM   = 0.355   # honest 3-fold CV on normalized yield
CV_R2_RAW    = 0.62    # equivalent raw yield R² (reported to users)
TEST_R2_NORM = 0.361
TEST_R2_RAW  = 0.62
N_SAMPLES    = 46_686
N_LOCATIONS  = 38
N_HYBRIDS    = 2_912
N_YEARS      = 5
MODEL_NAME   = "XGBoost"
DATASET      = "G2F 2014–2018"
FEATURE_NOTE = "Genomics (SNPs) + Weather + Plant Traits + Soil"

st.set_page_config(
    page_title="NeuroCrop — Maize Yield Predictor",
    page_icon="🌽",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  h1, h2, h3 { font-family: 'DM Serif Display', serif; }

  .hero-banner {
    background: linear-gradient(135deg, #0f4c2a 0%, #1a7a45 50%, #0d3d22 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    color: white;
  }
  .hero-banner h1 { color: white; font-size: 2.4rem; margin: 0 0 4px 0; }
  .hero-banner p  { color: #a8d5b5; margin: 0; font-size: 1rem; font-weight: 300; }

  .metric-card {
    background: #f8fdf9;
    border: 1px solid #d0e8d8;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-card .val { font-size: 1.8rem; font-weight: 600; color: #0f4c2a; font-family: 'DM Serif Display', serif; }
  .metric-card .lbl { font-size: 0.78rem; color: #5a7a66; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }

  .insight-box {
    background: #f0faf4;
    border-left: 4px solid #1a7a45;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.92rem;
    color: #1a3a25;
  }

  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-weight: 500;
    font-size: 0.88rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    # Canonical path: <repo>/outputs/predictions/all_predictions.csv.gz
    # Reads .csv.gz — pandas auto-detects gzip, no extra code needed
    csv_path = ROOT / "outputs" / "predictions" / "all_predictions.csv.gz"
    if not csv_path.exists():
        st.error(
            f"**all_predictions.csv.gz not found.**\n\n"
            f"Expected location: `outputs/predictions/all_predictions.csv.gz` "
            f"(relative to repo root).\n\n"
            f"Run Cell 37 in Colab to generate and compress the file, "
            f"then push outputs/predictions/all_predictions.csv.gz to GitHub."
        )
        st.stop()
    return pd.read_csv(csv_path)

df = load_data()

females   = sorted(df["Female"].unique().tolist())
males     = sorted(df["Male"].unique().tolist())
locations = sorted(df["Location"].unique().tolist())

# ── Helpers ────────────────────────────────────────────────────
@st.cache_data
def location_yields(loc: str) -> pd.Series:
    return df[df["Location"] == loc]["Yield"]

def get_percentile(yield_val: float, loc: str) -> float:
    sub = location_yields(loc)
    return round(float((sub < yield_val).mean()) * 100, 1) if len(sub) else 0.0

def category(y: float) -> str:
    if y >= 170: return "🟢 High"
    if y >= 150: return "🟡 Medium"
    return "🔴 Low"

def lookup(p1: str, p2: str, loc: str):
    res = df[(df["Female"] == p1) & (df["Male"] == p2) & (df["Location"] == loc)]
    if len(res) == 0:
        res = df[(df["Female"] == p2) & (df["Male"] == p1) & (df["Location"] == loc)]
    return round(float(res.iloc[0]["Yield"]), 2) if len(res) > 0 else None

@st.cache_data
def stability_table() -> pd.DataFrame:
    grp = df.groupby(["Female", "Male"])["Yield"]
    tbl = grp.agg(
        Mean_Yield=("mean"),
        Std_Yield=("std"),
        N_Locs=("count"),
    ).reset_index()
    tbl["CV_pct"]    = (tbl["Std_Yield"] / tbl["Mean_Yield"] * 100).round(1)
    tbl["Hybrid"]    = tbl["Female"] + " × " + tbl["Male"]
    tbl["Stability"] = tbl["CV_pct"].apply(
        lambda v: "🟢 Stable" if v < 5 else ("🟡 Moderate" if v < 10 else "🔴 Unstable")
    )
    return tbl.sort_values("Mean_Yield", ascending=False).reset_index(drop=True)

# ── PDF ────────────────────────────────────────────────────────
def generate_pdf_report(parent1, parent2, location, pred, loc_results, percentile):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("NeuroCrop — Maize Yield Prediction Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"Model: {MODEL_NAME}  |  Dataset: {DATASET}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Model Performance", styles["Heading2"]))
    story.append(Paragraph(
        f"CV R² (honest) = {CV_R2_NORM:.3f}  |  Test R² = {TEST_R2_NORM:.3f}  |  "
        f"Samples = {N_SAMPLES:,}  |  Locations = {N_LOCATIONS}  |  Years = {N_YEARS}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Prediction Summary", styles["Heading2"]))
    cat_clean = category(pred).replace("🟢","").replace("🟡","").replace("🔴","").strip()
    pred_data = [
        ["Parameter",       "Value"],
        ["Female Parent",   parent1],
        ["Male Parent",     parent2],
        ["Location",        location],
        ["Predicted Yield", f"{pred} bu/A"],
        ["Percentile Rank", f"Top {100 - percentile:.0f}% at this location"],
        ["Category",        cat_clean],
    ]
    tbl = Table(pred_data, colWidths=[200, 300])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1,  0), colors.HexColor("#0f4c2a")),
        ("TEXTCOLOR",      (0, 0), (-1,  0), colors.white),
        ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
        ("GRID",           (0, 0), (-1, -1), 1, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0faf4")]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 20))

    if loc_results:
        story.append(Paragraph("All Locations — Top 10", styles["Heading2"]))
        loc_data = [["Rank", "Location", "Yield (bu/A)", "Category"]]
        for i, row in enumerate(loc_results[:10], 1):
            cat_str = category(row["Yield"]).replace("🟢","").replace("🟡","").replace("🔴","").strip()
            loc_data.append([str(i), row["Location"], str(row["Yield"]), cat_str])
        loc_tbl = Table(loc_data, colWidths=[50, 150, 150, 150])
        loc_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1,  0), colors.HexColor("#0f4c2a")),
            ("TEXTCOLOR",      (0, 0), (-1,  0), colors.white),
            ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
            ("GRID",           (0, 0), (-1, -1), 1, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0faf4")]),
        ]))
        story.append(loc_tbl)

    story.append(Spacer(1, 30))
    story.append(Paragraph(
        f"NeuroCrop — Generative Breeding Platform  |  Abdul Manan  |  {DATASET}",
        styles["Normal"]
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
  <h1>🌽 NeuroCrop</h1>
  <p>Generative Breeding Platform · Maize Hybrid Yield Prediction · {DATASET} · {MODEL_NAME}</p>
</div>
""", unsafe_allow_html=True)

# Metric strip
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, val, lbl in [
    (c1, f"{CV_R2_NORM:.3f}",    "CV R² (honest)"),
    (c2, f"{TEST_R2_NORM:.3f}",  "Test R²"),
    (c3, f"{N_SAMPLES:,}",       "Training Samples"),
    (c4, f"{N_LOCATIONS}",       "Locations"),
    (c5, f"{N_HYBRIDS:,}",       "Hybrids"),
    (c6, f"{N_YEARS} yrs",       "Years (G2F)"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature importance callout
st.markdown("""
<div class="insight-box">
  <strong>Model inputs:</strong> Genomics (top 10k SNPs via mid-parent PCA) · 
  Season weather (May–Sep) · Critical-period weather (Jun–Aug) · 
  Soil properties · Plant morphology traits · 38 US field locations
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.markdown("## 🔬 Select Hybrid")
default_f = females.index("B73")  if "B73"  in females else 0
default_m = males.index("Mo17")   if "Mo17" in males   else 0

female   = st.sidebar.selectbox("Female Parent", females,   index=default_f)
male     = st.sidebar.selectbox("Male Parent",   males,     index=default_m)
location = st.sidebar.selectbox("Location",      locations)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Model:** {MODEL_NAME}  
**CV R²:** {CV_R2_NORM:.3f} (honest)  
**Test R²:** {TEST_R2_NORM:.3f}  
**Samples:** {N_SAMPLES:,}  
**Dataset:** {DATASET}  
""")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predict",
    "📍 Best Location",
    "🏆 Best Cross",
    "🔄 G×E Analysis",
    "📊 Stability",
    "📦 Batch Predict",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Single prediction
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Prediction: {female} × {male} @ {location}")
    pred = lookup(female, male, location)

    if pred:
        pct = get_percentile(pred, location)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Yield",  f"{pred} bu/A")
        m2.metric("Percentile Rank",  f"Top {100 - pct:.0f}%",
                  help="Rank vs all predicted crosses at this location")
        m3.metric("Female Parent",    female)
        m4.metric("Male Parent",      male)

        cat = category(pred)
        if   "High"   in cat: st.success(f"{cat} Yield")
        elif "Medium" in cat: st.warning(f"{cat} Yield")
        else:                 st.error(f"{cat} Yield")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            title={"text": "Predicted Yield (bu/A)", "font": {"size": 16}},
            delta={"reference": df["Yield"].mean(), "suffix": " bu/A vs avg"},
            gauge={
                "axis": {"range": [df["Yield"].min(), df["Yield"].max()]},
                "bar":  {"color": "#1a7a45"},
                "steps": [
                    {"range": [df["Yield"].min(), 150], "color": "#fde8e8"},
                    {"range": [150, 170],               "color": "#fef9c3"},
                    {"range": [170, df["Yield"].max()], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#0f4c2a", "width": 3},
                    "thickness": 0.8,
                    "value": df["Yield"].mean(),
                },
            }
        ))
        fig.update_layout(height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance context
        st.markdown("""
        <div class="insight-box">
          <strong>What drives this prediction:</strong>
          Genetics (PCA) 41% · Plant traits 24% · Season weather 19% · Critical-period weather 16%
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        @st.cache_data
        def _loc_results_for_pdf(p1, p2):
            rows = []
            for loc in locations:
                p = lookup(p1, p2, loc)
                if p:
                    rows.append({"Location": loc, "Yield": p})
            return sorted(rows, key=lambda x: x["Yield"], reverse=True)

        pdf_buf = generate_pdf_report(
            female, male, location, pred,
            _loc_results_for_pdf(female, male), pct
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buf,
            file_name=f"neurocrop_{female}_{male}_{location}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("Combination not found in database.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — Best location for a cross
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Best Locations for {female} × {male}")

    @st.cache_data
    def _best_locations(p1, p2):
        rows = []
        for loc in locations:
            p = lookup(p1, p2, loc)
            if p:
                rows.append({
                    "Location":   loc,
                    "Yield":      p,
                    "Percentile": get_percentile(p, loc),
                    "Category":   category(p),
                })
        return pd.DataFrame(rows).sort_values("Yield", ascending=False).reset_index(drop=True)

    res_df = _best_locations(female, male)

    if len(res_df):
        res_df.index += 1
        best = res_df.iloc[0]
        st.success(f"🏆 Best location: **{best['Location']}** → {best['Yield']} bu/A")

        fig = px.bar(
            res_df, x="Location", y="Yield",
            color="Yield", color_continuous_scale="RdYlGn",
            title=f"{female} × {male} — Yield by Location",
            text="Yield", template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, height=450, showlegend=False,
                          plot_bgcolor="#f8fdf9")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res_df, use_container_width=True)

        st.download_button(
            "📥 Download CSV",
            res_df.to_csv(index=False),
            f"best_locations_{female}_{male}.csv",
        )

# ══════════════════════════════════════════════════════════════
# TAB 3 — Best cross at a location
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Top Crosses at {location}")
    top_n = st.slider("Show top N crosses", 5, 50, 20)

    cross_df = df[df["Location"] == location].copy()
    cross_df["Cross"] = cross_df["Female"] + " × " + cross_df["Male"]
    loc_yields_s = cross_df["Yield"]
    cross_df["Percentile"] = cross_df["Yield"].apply(
        lambda v: round(float((loc_yields_s < v).mean()) * 100, 1)
    )
    cross_df = (cross_df
                .sort_values("Yield", ascending=False)
                .head(top_n)
                .reset_index(drop=True))
    cross_df.index += 1

    fig = px.bar(
        cross_df, x="Yield", y="Cross", orientation="h",
        color="Yield", color_continuous_scale="RdYlGn",
        title=f"Top {top_n} Crosses at {location}",
        text="Yield", template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        height=max(400, top_n * 25), showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="#f8fdf9",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cross_df[["Cross", "Yield", "Percentile"]], use_container_width=True)

    st.download_button(
        "📥 Download CSV",
        cross_df.to_csv(index=False),
        f"top_crosses_{location}.csv",
    )

# ══════════════════════════════════════════════════════════════
# TAB 4 — G×E interaction
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("G×E Interaction Analysis")
    st.markdown("""
    <div class="insight-box">
      G×E interaction = how the same hybrid responds differently across environments.
      Crossing lines in the chart = strong G×E (location-specific adaptation).
      Parallel lines = stable, wide-adapted hybrids.
    </div>""", unsafe_allow_html=True)

    sel_females = st.multiselect(
        "Select Female Parents", females, default=females[:3]
    )
    fixed_male = st.selectbox(
        "Fixed Male Parent", males,
        index=males.index("Mo17") if "Mo17" in males else 0,
        key="ge_male",
    )

    if sel_females:
        ge_df = df[
            df["Female"].isin(sel_females) & (df["Male"] == fixed_male)
        ].copy()
        ge_df["Hybrid"] = ge_df["Female"] + " × " + ge_df["Male"]

        if len(ge_df) > 0:
            fig = px.line(
                ge_df, x="Location", y="Yield", color="Hybrid",
                markers=True, title="G×E Interaction Across Locations",
                template="plotly_white",
            )
            fig.add_hline(
                y=ge_df["Yield"].mean(), line_dash="dash",
                line_color="#0f4c2a",
                annotation_text=f"Grand Mean ({ge_df['Yield'].mean():.1f} bu/A)",
            )
            fig.update_layout(height=500, xaxis_tickangle=-45,
                              plot_bgcolor="#f8fdf9")
            st.plotly_chart(fig, use_container_width=True)

            pivot = ge_df.pivot_table(
                index="Hybrid", columns="Location",
                values="Yield", aggfunc="mean"
            )
            if not pivot.empty:
                fig2 = px.imshow(
                    pivot, color_continuous_scale="RdYlGn",
                    title="Yield Heatmap (bu/A) — Genotype × Environment",
                    text_auto=".1f", template="plotly_white",
                )
                fig2.update_layout(height=max(300, len(sel_females) * 60 + 150))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for the selected combination.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — Stability ranking
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("G×E Stability Ranking")
    st.markdown("""
    <div class="insight-box">
      <strong>CV%</strong> = coefficient of variation across locations.
      Lower CV% = consistently high yield everywhere (wide adaptation).
      This is the core of <em>generative breeding</em>: identify crosses that are
      both high-yielding <em>and</em> stable — reducing the need for expensive multi-location trials.
    </div>""", unsafe_allow_html=True)

    stab = stability_table()

    c1, c2 = st.columns(2)
    min_mean = c1.slider("Min mean yield (bu/A)", int(df["Yield"].min()),
                         int(df["Yield"].max()), 150)
    max_cv   = c2.slider("Max CV% (stability threshold)", 1, 30, 10)

    filtered = stab[
        (stab["Mean_Yield"] >= min_mean) & (stab["CV_pct"] <= max_cv)
    ].head(50)

    if len(filtered):
        st.success(f"**{len(filtered)} hybrids** match: high yield + environmentally stable")

        fig = px.scatter(
            filtered, x="CV_pct", y="Mean_Yield",
            color="Stability", hover_data=["Hybrid", "N_Locs"],
            color_discrete_map={
                "🟢 Stable":   "#16a34a",
                "🟡 Moderate": "#ca8a04",
                "🔴 Unstable": "#dc2626",
            },
            title="Yield vs Stability — Ideal: top-right (high yield, low CV%)",
            labels={"CV_pct": "CV% across locations (lower = more stable)",
                    "Mean_Yield": "Mean Yield (bu/A)"},
            template="plotly_white",
        )
        fig.update_layout(height=450, plot_bgcolor="#f8fdf9")
        st.plotly_chart(fig, use_container_width=True)

        show_cols = ["Hybrid", "Mean_Yield", "Std_Yield", "CV_pct", "N_Locs", "Stability"]
        st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True)

        st.download_button(
            "📥 Download Stability Table",
            filtered[show_cols].to_csv(index=False),
            "neurocrop_stability_ranking.csv",
        )
    else:
        st.warning("No hybrids match the current filters. Try relaxing the thresholds.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — Batch prediction
# ══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📦 Batch Yield Prediction")
    st.markdown("Upload a CSV with columns: **Female, Male, Location**")

    sample = pd.DataFrame({
        "Female":   ["B73",  "A632",  "Oh43"],
        "Male":     ["Mo17", "Mo17",  "Mo17"],
        "Location": ["ILH1", "WIH1",  "IAH4"],
    })
    st.markdown("**Sample input format:**")
    st.dataframe(sample, use_container_width=True)
    st.download_button(
        "📥 Download Template CSV",
        sample.to_csv(index=False),
        "template.csv", "text/csv",
    )

    st.markdown("---")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded: {len(input_df)} rows**")
        st.dataframe(input_df.head(), use_container_width=True)

        missing_cols = [c for c in ["Female", "Male", "Location"]
                        if c not in input_df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            if st.button("🔮 Predict All", type="primary", use_container_width=True):
                results, errors = [], []
                progress = st.progress(0)
                status   = st.empty()
                total    = len(input_df)

                for i, row in input_df.iterrows():
                    p1  = str(row["Female"]).strip()
                    p2  = str(row["Male"]).strip()
                    loc = str(row["Location"]).strip()
                    p   = lookup(p1, p2, loc)

                    if p:
                        results.append({
                            "Female":          p1,
                            "Male":            p2,
                            "Location":        loc,
                            "Predicted Yield": p,
                            "Percentile":      get_percentile(p, loc),
                            "Category":        category(p),
                        })
                    else:
                        errors.append({"Female": p1, "Male": p2,
                                       "Location": loc, "Error": "Not found"})

                    progress.progress((i + 1) / total)
                    status.text(f"Processing {i + 1}/{total}...")

                progress.empty()
                status.empty()

                if results:
                    res_df = (pd.DataFrame(results)
                              .sort_values("Predicted Yield", ascending=False)
                              .reset_index(drop=True))
                    res_df.index += 1

                    st.success(f"✅ {len(results)} predictions completed")
                    if errors:
                        st.warning(f"⚠️ {len(errors)} rows not found in database")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Best Yield",    f"{res_df['Predicted Yield'].max():.1f} bu/A")
                    m2.metric("Average Yield", f"{res_df['Predicted Yield'].mean():.1f} bu/A")
                    m3.metric("Worst Yield",   f"{res_df['Predicted Yield'].min():.1f} bu/A")

                    st.dataframe(res_df, use_container_width=True)

                    fig = px.histogram(
                        res_df, x="Predicted Yield", nbins=20,
                        title="Yield Distribution — Batch Results",
                        color_discrete_sequence=["#1a7a45"],
                        template="plotly_white",
                    )
                    fig.update_layout(plot_bgcolor="#f8fdf9")
                    st.plotly_chart(fig, use_container_width=True)

                    c1, c2 = st.columns(2)
                    c1.download_button(
                        "📥 Download Results CSV",
                        res_df.to_csv(index=False),
                        "neurocrop_batch_predictions.csv", "text/csv",
                        use_container_width=True,
                    )
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        res_df.to_excel(writer, index=False, sheet_name="Predictions")
                    excel_buf.seek(0)
                    c2.download_button(
                        "📊 Download Results Excel",
                        excel_buf,
                        "neurocrop_batch_predictions.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                    if errors:
                        with st.expander(f"❌ {len(errors)} failed rows"):
                            st.dataframe(pd.DataFrame(errors), use_container_width=True)
                else:
                    st.error("No predictions found. Check Female/Male/Location values match the database.")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"**NeuroCrop** · Generative Breeding Platform · {DATASET} · {MODEL_NAME} · "
    f"CV R² = {CV_R2_NORM:.3f} · Test R² = {TEST_R2_NORM:.3f} · "
    f"{N_SAMPLES:,} samples · {N_LOCATIONS} locations · **Abdul Manan**"
)
