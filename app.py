"""Streamlit dashboard: Buyer Segmentation & Investment Profiling for Real Estate Market Intelligence."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Real Estate Buyer Intelligence", page_icon="🏢", layout="wide")

DATA = Path("data/processed/clients_segmented.csv")
METRICS = Path("data/processed/model_metrics.json")
COUNTRY_ALIAS = {"UAE": "United Arab Emirates", "USA": "United States", "UK": "United Kingdom"}


@st.cache_data
def load():
    df = pd.read_csv(DATA)
    df["country"] = df["country"].replace(COUNTRY_ALIAS)
    return df, json.load(open(METRICS))


if not DATA.exists():
    st.error("Processed data not found. Open Command Prompt in the project folder and run:  python run_pipeline.py")
    st.stop()

df_all, metrics = load()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏢 Buyer Intelligence")
page = st.sidebar.radio("Navigate", ["Buyer Segmentation Overview", "Investor Behavior Dashboard",
                                     "Geographic Buyer Analysis", "Segment Insights Panel"])
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
types = st.sidebar.multiselect("Client type", sorted(df_all.client_type.unique()), default=sorted(df_all.client_type.unique()))
regions = st.sidebar.multiselect("Region", sorted(df_all.region.unique()), default=sorted(df_all.region.unique()))
df = df_all[df_all.client_type.isin(types) & df_all.region.isin(regions)]
if df.empty:
    st.warning("No clients match the selected filters.")
    st.stop()
st.sidebar.caption(f"{len(df):,} of {len(df_all):,} clients shown")

seg_order = df.segment.value_counts().index.tolist()
SEG_COLORS = dict(zip(sorted(df_all.segment.unique()), px.colors.qualitative.Bold))
cat_orders = {"segment": seg_order}


def pct(s):
    return f"{s.mean() * 100:.1f}%"


# ================================================================== 1. OVERVIEW
if page == "Buyer Segmentation Overview":
    st.title("Buyer Segmentation Overview")
    st.caption("Machine-learning (K-Means) segmentation of real estate buyers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total clients", f"{len(df):,}")
    c2.metric("Segments found", df.segment.nunique())
    c3.metric("Investor share", pct(df.is_investor))
    c4.metric("Avg satisfaction", f"{df.satisfaction_score.mean():.2f} / 5")

    left, right = st.columns(2)
    counts = df.segment.value_counts().reset_index()
    counts.columns = ["segment", "clients"]
    with left:
        fig = px.pie(counts, names="segment", values="clients", hole=.45, color="segment",
                     color_discrete_map=SEG_COLORS, title="Cluster distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(counts, x="clients", y="segment", orientation="h", color="segment", text="clients",
                     color_discrete_map=SEG_COLORS, title="Clients per segment")
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment map (2-D projection of the model)")
    sample = df.sample(min(2000, len(df)), random_state=1)
    fig = px.scatter(sample, x="pca_1", y="pca_2", color="segment", color_discrete_map=SEG_COLORS,
                     hover_data=["client_type", "country", "acquisition_purpose"], opacity=.7,
                     category_orders=cat_orders)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"K chosen by silhouette score: k = {metrics['chosen_k']} "
               f"(score {metrics['silhouette_by_k'][str(metrics['chosen_k'])]:.2f}). "
               f"The 2-D view captures {metrics['explained_variance_2d'] * 100:.0f}% of variance.")

# ================================================================== 2. INVESTOR BEHAVIOR
elif page == "Investor Behavior Dashboard":
    st.title("Investor Behavior Dashboard")
    g = df.groupby("segment").agg(clients=("client_id", "count"), investment_rate=("is_investor", "mean"),
                                  loan_rate=("has_loan", "mean"), satisfaction=("satisfaction_score", "mean")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        d = df.groupby(["segment", "acquisition_purpose"]).size().reset_index(name="clients")
        st.plotly_chart(px.bar(d, x="segment", y="clients", color="acquisition_purpose", barmode="stack",
                               title="Acquisition purpose by segment", category_orders=cat_orders),
                        use_container_width=True)
    with c2:
        d = df.groupby(["segment", "loan_applied"]).size().reset_index(name="clients")
        st.plotly_chart(px.bar(d, x="segment", y="clients", color="loan_applied", barmode="group",
                               title="Financing (loan applied) by segment", category_orders=cat_orders),
                        use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        d = df.groupby(["segment", "referral_channel"]).size().reset_index(name="clients")
        st.plotly_chart(px.density_heatmap(d, x="referral_channel", y="segment", z="clients", text_auto=True,
                                           color_continuous_scale="Blues", title="Referral channel × segment"),
                        use_container_width=True)
    with c4:
        st.plotly_chart(px.box(df, x="segment", y="satisfaction_score", color="segment",
                               color_discrete_map=SEG_COLORS, title="Satisfaction by segment",
                               category_orders=cat_orders).update_layout(showlegend=False),
                        use_container_width=True)
    d = df.groupby(["age_group", "segment"]).size().reset_index(name="clients")
    st.plotly_chart(px.bar(d, x="age_group", y="clients", color="segment", color_discrete_map=SEG_COLORS,
                           title="Age profile of segments",
                           category_orders={"age_group": ["<30", "30-39", "40-49", "50-59", "60+"]}),
                    use_container_width=True)
    show = g.copy()
    show["investment_rate"] = (show.investment_rate * 100).round(1).astype(str) + "%"
    show["loan_rate"] = (show.loan_rate * 100).round(1).astype(str) + "%"
    show["satisfaction"] = show.satisfaction.round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)

# ================================================================== 3. GEOGRAPHY
elif page == "Geographic Buyer Analysis":
    st.title("Geographic Buyer Analysis")
    view = st.radio("Map metric", ["Total buyers", "Investor share", "Dominant segment"], horizontal=True)
    geo = df.groupby("country").agg(clients=("client_id", "count"), investor_share=("is_investor", "mean"),
                                    loan_rate=("has_loan", "mean"),
                                    top_segment=("segment", lambda s: s.value_counts().index[0])).reset_index()
    if view == "Total buyers":
        fig = px.choropleth(geo, locations="country", locationmode="country names", color="clients",
                            color_continuous_scale="Blues", hover_data=["top_segment"])
    elif view == "Investor share":
        fig = px.choropleth(geo, locations="country", locationmode="country names", color="investor_share",
                            color_continuous_scale="Oranges", hover_data=["clients", "top_segment"])
    else:
        fig = px.choropleth(geo, locations="country", locationmode="country names", color="top_segment",
                            color_discrete_map=SEG_COLORS, hover_data=["clients"])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=480)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        d = df.groupby(["region", "segment"]).size().reset_index(name="clients")
        st.plotly_chart(px.bar(d, x="region", y="clients", color="segment", color_discrete_map=SEG_COLORS,
                               title="Segments by region", barmode="stack"), use_container_width=True)
    with c2:
        r = df.groupby("region").agg(investor_share=("is_investor", "mean"), loan_rate=("has_loan", "mean")).reset_index()
        r = r.melt("region", var_name="metric", value_name="share")
        st.plotly_chart(px.bar(r, x="region", y="share", color="metric", barmode="group",
                               title="Investment & financing behaviour by region").update_yaxes(tickformat=".0%"),
                        use_container_width=True)
    mix = pd.crosstab(df.region, df.segment, normalize="index").round(3) * 100
    st.subheader("Segment mix per region (%)")
    st.dataframe(mix.round(1), use_container_width=True)

# ================================================================== 4. INSIGHTS
else:
    st.title("Segment Insights Panel")
    seg = st.selectbox("Choose a segment", seg_order)
    s = df[df.segment == seg]
    share = len(s) / len(df) * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clients", f"{len(s):,}", f"{share:.1f}% of total")
    c2.metric("Avg age", f"{s.age.mean():.0f}")
    c3.metric("Investors", pct(s.is_investor))
    c4.metric("Loan applied", pct(s.has_loan))
    c5.metric("Satisfaction", f"{s.satisfaction_score.mean():.2f}")

    top = lambda col: s[col].value_counts().index[0]
    st.info(f"**{seg}** — {share:.1f}% of clients. Typical buyer: {top('client_type').lower()}, "
            f"average age {s.age.mean():.0f}, mostly from **{top('country')}** ({top('region')}), "
            f"acquired mainly via **{top('referral_channel')}**. "
            f"{s.is_investor.mean() * 100:.0f}% buy for investment and {s.has_loan.mean() * 100:.0f}% use financing.")

    st.subheader("Descriptive statistics per cluster")
    stats = df.groupby("segment").agg(
        clients=("client_id", "count"), age_mean=("age", "mean"), age_median=("age", "median"),
        age_std=("age", "std"), satisfaction_mean=("satisfaction_score", "mean"),
        satisfaction_std=("satisfaction_score", "std"), investor_pct=("is_investor", "mean"),
        loan_pct=("has_loan", "mean"), corporate_pct=("is_corporate", "mean")).round(2)
    for c in ["investor_pct", "loan_pct", "corporate_pct"]:
        stats[c] = (stats[c] * 100).round(1)
    st.dataframe(stats, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(s, x="age", nbins=20, title=f"Age distribution — {seg}"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(s, names="referral_channel", title="Acquisition channels", hole=.4), use_container_width=True)
    st.subheader("Top countries")
    st.dataframe(s.country.value_counts().head(5).rename("clients").to_frame(), use_container_width=True)
    st.download_button("⬇ Download segmented data (CSV)", df.to_csv(index=False).encode(),
                       "segmented_clients.csv", "text/csv")
