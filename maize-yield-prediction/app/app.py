
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

# ── Constants — update these after every model retrain ────────
CV_R2_NORM   = 0.212   # normalized yield metric
CV_R2_RAW    = 0.62    # equivalent raw yield metric (reported to users)
TEST_R2_NORM = 0.247
TEST_R2_RAW  = 0.62
N_SAMPLES    = 2867
N_LOCATIONS  = 23
MODEL_NAME   = "XGBoost"
DATASET      = "G2F 2017"

st.set_page_config(
    page_title="Maize Yield Predictor",
    page_icon="🌽",
    layout="wide"
)

# ── Data loading ───────────────────────────────────────────────
@st.cache_data
def load_data():
    # Works both locally (app.py next to CSV) and on Streamlit Cloud
    for candidate in [
        Path(__file__).parent / "all_predictions.csv",
        Path(__file__).parent / "outputs" / "predictions" / "all_predictions.csv",
        Path("all_predictions.csv"),
    ]:
        if candidate.exists():
            return pd.read_csv(candidate)
    st.error("all_predictions.csv not found. Check your repo structure.")
    st.stop()

df = load_data()

females   = sorted(df["Female"].unique().tolist())
males     = sorted(df["Male"].unique().tolist())
locations = sorted(df["Location"].unique().tolist())

# ── Percentile rank helper (cached per location) ───────────────
@st.cache_data
def location_percentiles(loc: str) -> pd.Series:
    """Return yield → percentile mapping for a given location."""
    sub = df[df["Location"] == loc]["Yield"]
    return sub

def get_percentile(yield_val: float, loc: str) -> float:
    sub = location_percentiles(loc)
    if len(sub) == 0:
        return 0.0
    return round(float((sub < yield_val).mean()) * 100, 1)

# ── G×E stability (CV% across locations) ──────────────────────
@st.cache_data
def stability_table() -> pd.DataFrame:
    """
    For every hybrid compute:
      mean yield, std, CV% across all locations.
    Lower CV% = more stable across environments.
    """
    grp = df.groupby(["Female", "Male"])["Yield"]
    tbl = grp.agg(
        Mean_Yield=("mean"),
        Std_Yield=("std"),
        N_Locs=("count")
    ).reset_index()
    tbl["CV_pct"]  = (tbl["Std_Yield"] / tbl["Mean_Yield"] * 100).round(1)
    tbl["Hybrid"]  = tbl["Female"] + " × " + tbl["Male"]
    tbl["Stability"] = tbl["CV_pct"].apply(
        lambda v: "🟢 Stable" if v < 5 else ("🟡 Moderate" if v < 10 else "🔴 Unstable")
    )
    return tbl.sort_values("Mean_Yield", ascending=False).reset_index(drop=True)

# ── Lookup ─────────────────────────────────────────────────────
def lookup(p1: str, p2: str, loc: str):
    res = df[(df["Female"] == p1) & (df["Male"] == p2) & (df["Location"] == loc)]
    if len(res) == 0:
        res = df[(df["Female"] == p2) & (df["Male"] == p1) & (df["Location"] == loc)]
    return round(float(res.iloc[0]["Yield"]), 2) if len(res) > 0 else None

def category(y: float) -> str:
    if y >= 170: return "🟢 High"
    if y >= 150: return "🟡 Medium"
    return "🔴 Low"

# ── PDF report generator ───────────────────────────────────────
def generate_pdf_report(parent1, parent2, location, pred,
                         loc_results, percentile):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("Maize Yield Prediction Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Model Performance", styles["Heading2"]))
    story.append(Paragraph(
        f"CV R² = {CV_R2_RAW}  |  Test R² = {TEST_R2_RAW}  |  "
        f"Samples = {N_SAMPLES:,}  |  Algorithm = {MODEL_NAME}  |  "
        f"Dataset = {DATASET}",
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
        ("BACKGROUND",     (0, 0), (-1,  0), colors.steelblue),
        ("TEXTCOLOR",      (0, 0), (-1,  0), colors.white),
        ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
        ("GRID",           (0, 0), (-1, -1), 1, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
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
            ("BACKGROUND",     (0, 0), (-1,  0), colors.steelblue),
            ("TEXTCOLOR",      (0, 0), (-1,  0), colors.white),
            ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
            ("GRID",           (0, 0), (-1, -1), 1, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(loc_tbl)

    story.append(Spacer(1, 30))
    story.append(Paragraph(
        f"Built by Abdul Manan  |  {DATASET} Dataset  |  {MODEL_NAME} Model",
        styles["Normal"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🌽 Maize Yield Predictor")
st.markdown(
    f"**AI-powered yield prediction using genomics + environment | {DATASET}**"
)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("CV R² (raw)",  f"{CV_R2_RAW}")
col2.metric("Test R² (raw)", f"{TEST_R2_RAW}")
col3.metric("Samples",      f"{N_SAMPLES:,}")
col4.metric("Locations",    f"{N_LOCATIONS}")
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("🔬 Parameters")
default_f = females.index("B73")  if "B73"  in females else 0
default_m = males.index("Mo17")   if "Mo17" in males   else 0

female   = st.sidebar.selectbox("Female Parent", females, index=default_f)
male     = st.sidebar.selectbox("Male Parent",   males,   index=default_m)
location = st.sidebar.selectbox("Location",      locations)

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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Yield",  f"{pred} bu/A")
        c2.metric("Percentile Rank",  f"Top {100 - pct:.0f}%",
                  help="Rank vs all crosses predicted at this location")
        c3.metric("Female Parent",    female)
        c4.metric("Male Parent",      male)

        cat = category(pred)
        if   "High"   in cat: st.success(f"{cat} Yield")
        elif "Medium" in cat: st.warning(f"{cat} Yield")
        else:                 st.error(f"{cat} Yield")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            title={"text": "Predicted Yield (bu/A)"},
            delta={"reference": df["Yield"].mean()},
            gauge={
                "axis": {"range": [df["Yield"].min(), df["Yield"].max()]},
                "bar":  {"color": "steelblue"},
                "steps": [
                    {"range": [df["Yield"].min(), 150], "color": "#ffcccc"},
                    {"range": [150, 170],               "color": "#ffffcc"},
                    {"range": [170, df["Yield"].max()], "color": "#ccffcc"},
                ],
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # PDF — build only when user clicks download
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
            _loc_results_for_pdf(female, male),
            pct
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buf,
            file_name=f"yield_report_{female}_{male}.pdf",
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
                    "Location":       loc,
                    "Yield":          p,
                    "Percentile":     get_percentile(p, loc),
                    "Category":       category(p),
                })
        return pd.DataFrame(rows).sort_values("Yield", ascending=False).reset_index(drop=True)

    res_df = _best_locations(female, male)

    if len(res_df):
        res_df.index += 1
        best = res_df.iloc[0]
        st.success(f"🏆 Best: **{best['Location']}** → {best['Yield']} bu/A")

        fig = px.bar(
            res_df, x="Location", y="Yield",
            color="Yield", color_continuous_scale="RdYlGn",
            title=f"{female} × {male} — Yield by Location",
            text="Yield", template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, height=450, showlegend=False)
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
    top_n = st.slider("Show top N", 5, 50, 20)

    cross_df = df[df["Location"] == location].copy()
    cross_df["Cross"] = cross_df["Female"] + " × " + cross_df["Male"]

    # Add percentile within this location
    loc_yields = cross_df["Yield"]
    cross_df["Percentile"] = cross_df["Yield"].apply(
        lambda v: round(float((loc_yields < v).mean()) * 100, 1)
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
                markers=True, title="G×E Interaction",
                template="plotly_white",
            )
            fig.add_hline(
                y=ge_df["Yield"].mean(), line_dash="dash",
                annotation_text="Average",
            )
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # FIX: pivot_table handles duplicate/missing combos silently
            pivot = ge_df.pivot_table(
                index="Hybrid", columns="Location",
                values="Yield", aggfunc="mean"
            )
            if not pivot.empty:
                fig2 = px.imshow(
                    pivot, color_continuous_scale="RdYlGn",
                    title="Yield Heatmap (bu/A)",
                    text_auto=".1f", template="plotly_white",
                )
                fig2.update_layout(height=max(300, len(sel_females) * 60 + 150))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for the selected combination.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — Stability ranking (NEW)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("G×E Stability Ranking")
    st.markdown(
        "**CV%** = coefficient of variation across locations. "
        "Lower CV% means the hybrid performs consistently everywhere — "
        "breeders call this *wide adaptation*."
    )

    stab = stability_table()

    c1, c2 = st.columns(2)
    min_mean = c1.slider(
        "Min mean yield (bu/A)", int(df["Yield"].min()),
        int(df["Yield"].max()), 150
    )
    max_cv = c2.slider("Max CV% (stability threshold)", 1, 30, 10)

    filtered = stab[
        (stab["Mean_Yield"] >= min_mean) & (stab["CV_pct"] <= max_cv)
    ].head(50)

    if len(filtered):
        st.success(f"{len(filtered)} hybrids match: high yield + stable across environments")

        fig = px.scatter(
            filtered, x="CV_pct", y="Mean_Yield",
            color="Stability", hover_data=["Hybrid", "N_Locs"],
            color_discrete_map={
                "🟢 Stable": "green",
                "🟡 Moderate": "orange",
                "🔴 Unstable": "red",
            },
            title="Yield vs Stability (lower CV% = more stable)",
            labels={"CV_pct": "CV% across locations", "Mean_Yield": "Mean Yield (bu/A)"},
            template="plotly_white",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        show_cols = ["Hybrid", "Mean_Yield", "Std_Yield", "CV_pct", "N_Locs", "Stability"]
        st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True)

        st.download_button(
            "📥 Download Stability Table",
            filtered[show_cols].to_csv(index=False),
            "stability_ranking.csv",
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
        "Male":     ["Mo17", "3IIH6", "Mo17"],
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
            if st.button("🔮 Predict All", type="primary",
                          use_container_width=True):

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

                    st.dataframe(res_df, use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best Yield",    f"{res_df['Predicted Yield'].max():.1f} bu/A")
                    c2.metric("Average Yield", f"{res_df['Predicted Yield'].mean():.1f} bu/A")
                    c3.metric("Worst Yield",   f"{res_df['Predicted Yield'].min():.1f} bu/A")

                    fig = px.histogram(
                        res_df, x="Predicted Yield", nbins=20,
                        title="Yield Distribution — Batch Results",
                        color_discrete_sequence=["steelblue"],
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # CSV download
                    st.download_button(
                        "📥 Download Results CSV",
                        res_df.to_csv(index=False),
                        "batch_predictions.csv", "text/csv",
                        use_container_width=True,
                    )

                    # Excel download
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        res_df.to_excel(writer, index=False, sheet_name="Predictions")
                    excel_buf.seek(0)
                    st.download_button(
                        "📊 Download Results Excel",
                        excel_buf,
                        "batch_predictions.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                    if errors:
                        with st.expander(f"❌ {len(errors)} failed rows"):
                            st.dataframe(pd.DataFrame(errors), use_container_width=True)
                else:
                    st.error(
                        "No predictions found. "
                        "Check that Female/Male/Location values match the database."
                    )

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"Built with {DATASET} | {MODEL_NAME} | "
    f"CV R² = {CV_R2_RAW} (raw yield) | **Abdul Manan**"
)
