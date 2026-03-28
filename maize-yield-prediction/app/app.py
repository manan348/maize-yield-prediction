
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import io
import datetime
from pathlib import Path

# ── PDF imports ────────────────────────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles    import getSampleStyleSheet
from reportlab.platypus      import (SimpleDocTemplate,
                                      Paragraph, Spacer,
                                      Table, TableStyle)
from reportlab.lib           import colors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.predict import lookup_precomputed_prediction

st.set_page_config(
    page_title="Maize Yield Predictor",
    page_icon="🌽",
    layout="wide"
)

@st.cache_data
def load_data():
    preds_path = ROOT / "outputs" / "predictions" / "all_predictions.csv"
    return pd.read_csv(preds_path)

df = load_data()

# Get unique values
females   = sorted(df["Female"].unique().tolist())
males     = sorted(df["Male"].unique().tolist())
locations = sorted(df["Location"].unique().tolist())

def lookup(p1, p2, loc):
    return lookup_precomputed_prediction(df, p1, p2, loc)

def category(y):
    if y >= 170: return "🟢 High"
    if y >= 150: return "🟡 Medium"
    return "🔴 Low"

# ── PDF Report Generator ───────────────────────────────
def generate_pdf_report(parent1, parent2,
                         location, pred, loc_results):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    story.append(Paragraph(
        "Maize Yield Prediction Report",
        styles['Title']
    ))
    story.append(Spacer(1, 12))

    # Date
    story.append(Paragraph(
        f"Generated: "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))

    # Model info
    story.append(Paragraph(
        "Model Performance",
        styles['Heading2']
    ))
    story.append(Paragraph(
        "CV R² = 0.572  |  Test R² = 0.635  |  "
        "Samples = 2,867  |  Algorithm = Random Forest",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))

    # Prediction summary table
    story.append(Paragraph(
        "Prediction Summary",
        styles['Heading2']
    ))

    pred_data = [
        ['Parameter',      'Value'],
        ['Female Parent',  parent1],
        ['Male Parent',    parent2],
        ['Location',       location],
        ['Predicted Yield',f'{pred} bu/A'],
        ['Category',       category(pred).replace("🟢","").replace("🟡","").replace("🔴","").strip()]
    ]

    table = Table(pred_data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), colors.steelblue),
        ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('GRID',          (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1),
         [colors.white, colors.lightgrey])
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Location comparison table
    if loc_results:
        story.append(Paragraph(
            "All Locations Comparison (Top 10)",
            styles['Heading2']
        ))

        loc_data = [['Rank', 'Location', 'Yield (bu/A)', 'Category']]
        for i, row in enumerate(loc_results[:10], 1):
            loc_data.append([
                str(i),
                row['Location'],
                str(row['Yield']),
                category(row['Yield']).replace("🟢","").replace("🟡","").replace("🔴","").strip()
            ])

        loc_table = Table(loc_data, colWidths=[50, 150, 150, 150])
        loc_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0), colors.steelblue),
            ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
            ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('GRID',          (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1),
             [colors.white, colors.lightgrey])
        ]))
        story.append(loc_table)

    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "Built by Abdul Manan  |  G2F 2017 Dataset  |  Random Forest Model",
        styles['Normal']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Header ─────────────────────────────────────────────
st.title("🌽 Maize Yield Predictor")
st.markdown(
    "**AI-powered yield prediction using "
    "genomics + environment | G2F 2017**"
)
st.markdown("---")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("CV R²",     "0.572")
col2.metric("Test R²",   "0.635")
col3.metric("Samples",   "2,867")
col4.metric("Locations", "23")
st.markdown("---")

# Sidebar inputs
st.sidebar.title("🔬 Parameters")
default_f = females.index("B73")  if "B73"  in females else 0
default_m = males.index("Mo17")   if "Mo17" in males   else 0

female   = st.sidebar.selectbox("Female Parent",
                                  females, index=default_f)
male     = st.sidebar.selectbox("Male Parent",
                                  males,   index=default_m)
location = st.sidebar.selectbox("Location", locations)

# ── Tabs — added tab_batch as 5th tab ─────────────────
tab1, tab2, tab3, tab4, tab_batch = st.tabs([
    "🔮 Predict",
    "📍 Best Location",
    "🏆 Best Cross",
    "🔄 G×E Analysis",
    "📦 Batch Predict"
])

# ── TAB 1: Single Prediction + PDF Download ────────────
with tab1:
    st.subheader(
        f"Prediction: {female} × {male} @ {location}"
    )
    pred = lookup(female, male, location)

    if pred:
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Yield", f"{pred} bu/A")
        col2.metric("Female Parent",   female)
        col3.metric("Male Parent",     male)

        cat = category(pred)
        if "High"   in cat: st.success(f"{cat} Yield")
        elif "Medium" in cat: st.warning(f"{cat} Yield")
        else:                 st.error(f"{cat} Yield")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            title={"text": "Predicted Yield (bu/A)"},
            delta={"reference": df["Yield"].mean()},
            gauge={
                "axis": {
                    "range": [
                        df["Yield"].min(),
                        df["Yield"].max()
                    ]
                },
                "bar": {"color": "steelblue"},
                "steps": [
                    {"range": [df["Yield"].min(), 150],
                     "color": "#ffcccc"},
                    {"range": [150, 170],
                     "color": "#ffffcc"},
                    {"range": [170, df["Yield"].max()],
                     "color": "#ccffcc"}
                ]
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # ── PDF Download Button ────────────────────────
        st.markdown("---")
        loc_results = []
        for loc in locations:
            p = lookup(female, male, loc)
            if p:
                loc_results.append({
                    'Location': loc,
                    'Yield':    p
                })
        loc_results.sort(
            key=lambda x: x['Yield'], reverse=True
        )
        pdf_buffer = generate_pdf_report(
            female, male, location, pred, loc_results
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"yield_report_{female}_{male}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        # ──────────────────────────────────────────────

    else:
        st.warning("Combination not found in database")

# ── TAB 2: Best Location ───────────────────────────────
with tab2:
    st.subheader(f"Best Locations for {female} × {male}")
    rows = []
    for loc in locations:
        p = lookup(female, male, loc)
        if p:
            rows.append({
                "Location": loc,
                "Yield":    p,
                "Category": category(p)
            })

    if rows:
        res_df = pd.DataFrame(rows)\
                   .sort_values("Yield", ascending=False)\
                   .reset_index(drop=True)
        res_df.index += 1

        best = res_df.iloc[0]
        st.success(
            f"🏆 Best: **{best['Location']}** "
            f"→ {best['Yield']} bu/A"
        )

        fig = px.bar(
            res_df,
            x="Location", y="Yield",
            color="Yield",
            color_continuous_scale="RdYlGn",
            title=f"{female} × {male} — Yield by Location",
            text="Yield",
            template="plotly_white"
        )
        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=450,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res_df, use_container_width=True)

        csv = res_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV", csv,
            f"best_locations_{female}_{male}.csv"
        )

# ── TAB 3: Best Cross at Location ─────────────────────
with tab3:
    st.subheader(f"Top Crosses at {location}")
    top_n = st.slider("Show top N", 5, 50, 20)

    cross_df = df[df["Location"] == location].copy()
    cross_df["Cross"] = (cross_df["Female"] +
                          " × " + cross_df["Male"])
    cross_df = cross_df\
               .sort_values("Yield", ascending=False)\
               .head(top_n)\
               .reset_index(drop=True)
    cross_df.index += 1

    fig = px.bar(
        cross_df,
        x="Yield", y="Cross",
        orientation="h",
        color="Yield",
        color_continuous_scale="RdYlGn",
        title=f"Top {top_n} Crosses at {location}",
        text="Yield",
        template="plotly_white"
    )
    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )
    fig.update_layout(
        height=max(400, top_n*25),
        showlegend=False,
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True)

    csv = cross_df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV", csv,
        f"top_crosses_{location}.csv"
    )

# ── TAB 4: G×E Analysis ───────────────────────────────
with tab4:
    st.subheader("G×E Interaction Analysis")

    sel_females = st.multiselect(
        "Select Female Parents",
        females,
        default=females[:3]
    )
    fixed_male = st.selectbox(
        "Fixed Male Parent", males,
        index=males.index("Mo17") if "Mo17" in males else 0
    )

    if sel_females:
        ge_df = df[
            (df["Female"].isin(sel_females)) &
            (df["Male"] == fixed_male)
        ].copy()
        ge_df["Hybrid"] = (ge_df["Female"] +
                            " × " + ge_df["Male"])

        if len(ge_df) > 0:
            fig = px.line(
                ge_df,
                x="Location", y="Yield",
                color="Hybrid",
                markers=True,
                title="G×E Interaction",
                template="plotly_white"
            )
            fig.add_hline(
                y=ge_df["Yield"].mean(),
                line_dash="dash",
                annotation_text="Average"
            )
            fig.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

            pivot = ge_df.pivot(
                index="Hybrid",
                columns="Location",
                values="Yield"
            )
            fig2 = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                title="Yield Heatmap",
                text_auto=".1f",
                template="plotly_white"
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

# ── TAB 5: Batch Predict ──────────────────────────────
with tab_batch:
    st.subheader("📦 Batch Yield Prediction")
    st.markdown(
        "Upload a CSV with columns: **Female, Male, Location**"
    )

    # Show sample format
    sample = pd.DataFrame({
        'Female':   ['B73', 'A632', 'Oh43'],
        'Male':     ['Mo17', '3IIH6', 'Mo17'],
        'Location': ['ILH1', 'WIH1', 'IAH4']
    })
    st.markdown("**Sample input format:**")
    st.dataframe(sample, use_container_width=True)

    # Download template
    st.download_button(
        "📥 Download Template CSV",
        sample.to_csv(index=False),
        "template.csv",
        "text/csv"
    )

    st.markdown("---")

    # Upload CSV
    uploaded = st.file_uploader(
        "Upload your CSV file",
        type=['csv']
    )

    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded: {len(input_df)} rows**")
        st.dataframe(input_df.head(), use_container_width=True)

        # Validate columns
        required = ['Female', 'Male', 'Location']
        missing  = [c for c in required
                    if c not in input_df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.info("Required columns: Female, Male, Location")
        else:
            if st.button("🔮 Predict All",
                          type="primary",
                          use_container_width=True):

                results  = []
                errors   = []
                progress = st.progress(0)
                status   = st.empty()
                total    = len(input_df)

                for i, row in input_df.iterrows():
                    p1  = str(row['Female']).strip()
                    p2  = str(row['Male']).strip()
                    loc = str(row['Location']).strip()

                    p = lookup(p1, p2, loc)

                    if p:
                        results.append({
                            'Female':          p1,
                            'Male':            p2,
                            'Location':        loc,
                            'Predicted Yield': p,
                            'Category':        category(p)
                        })
                    else:
                        errors.append({
                            'Female':   p1,
                            'Male':     p2,
                            'Location': loc,
                            'Error':    'Not found in database'
                        })

                    progress.progress((i + 1) / total)
                    status.text(f"Processing {i+1}/{total}...")

                progress.empty()
                status.empty()

                if results:
                    res_df = pd.DataFrame(results)\
                               .sort_values(
                                   'Predicted Yield',
                                   ascending=False
                               ).reset_index(drop=True)
                    res_df.index += 1

                    st.success(
                        f"✅ {len(results)} predictions completed"
                    )
                    if errors:
                        st.warning(
                            f"⚠️ {len(errors)} rows not found in database"
                        )

                    # Results table
                    st.dataframe(res_df, use_container_width=True)

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Best Yield",
                        f"{res_df['Predicted Yield'].max():.1f} bu/A"
                    )
                    col2.metric(
                        "Average Yield",
                        f"{res_df['Predicted Yield'].mean():.1f} bu/A"
                    )
                    col3.metric(
                        "Worst Yield",
                        f"{res_df['Predicted Yield'].min():.1f} bu/A"
                    )

                    # Distribution chart
                    fig = px.histogram(
                        res_df,
                        x='Predicted Yield',
                        nbins=20,
                        title='Yield Distribution of Batch Results',
                        color_discrete_sequence=['steelblue'],
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Download results
                    st.download_button(
                        "📥 Download Results CSV",
                        res_df.to_csv(index=False),
                        "batch_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )

                    # Show failed rows
                    if errors:
                        with st.expander(
                            f"❌ {len(errors)} failed rows — click to view"
                        ):
                            st.dataframe(
                                pd.DataFrame(errors),
                                use_container_width=True
                            )
                else:
                    st.error(
                        "❌ No predictions found. "
                        "Check that your Female/Male/Location "
                        "values match the database."
                    )

# ── Footer ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Built with G2F 2017 | Random Forest | "
    "CV R² = 0.572 | **Abdul Manan**"
)
