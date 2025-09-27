import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Bakeries CA — BI Dashboard", layout="wide")

# -----------------------------
# Data loading and preparation
# -----------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    needed = ["name","address","latitude","longitude","category","avg_rating",
              "num_of_reviews","price","state","url","main_category"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")

    df = df.dropna(subset=["latitude","longitude"]).copy()

    if "city" not in df.columns:
        import re
        def extract_city(addr):
            if pd.isna(addr): return "Unknown"
            m = re.search(r",\s*([^,]+),\s*CA\b", str(addr))
            return m.group(1).strip() if m else "Unknown"
        df["city"] = df["address"].apply(extract_city)

    price_ord_map = {"low":0, "medium":1, "high":2, "unknown":1}
    df["price_ordinal"] = df["price"].map(price_ord_map).fillna(1).astype(float)
    df["log_reviews"] = np.log1p(df["num_of_reviews"])
    return df

df = load_data("bakery_data_clean.csv")

# -----------------------------
# Filter 
# -----------------------------
st.sidebar.header("Filters")
city_all = ["(All)"] + sorted(df["city"].dropna().unique().tolist())
city_sel = st.sidebar.selectbox("Cities", city_all, index=0)

df_f = df.copy()
if city_sel != "(All)":
    df_f = df_f[df_f["city"] == city_sel].copy()

X_all = df[["latitude","longitude","avg_rating","log_reviews","price_ordinal"]].copy()

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_all = kmeans.fit_predict(X_all_scaled)

# Centroids in ORIGINAL feature space to decide south/north + demand
centroids_scaled = kmeans.cluster_centers_
centroids = pd.DataFrame(
    scaler.inverse_transform(centroids_scaled),
    columns=["latitude","longitude","avg_rating","log_reviews","price_ordinal"]
)
centroids["orig_label"] = range(5)

# --- Canonical remap rules ---
# Sort by latitude: south -> north
centroids_sorted = centroids.sort_values("latitude")

south_two = centroids_sorted.iloc[:2].copy()
middle_one = centroids_sorted.iloc[2:3].copy()
north_two  = centroids_sorted.iloc[3:].copy()

# Use log_reviews as a proxy for demand/popularity
south_low  = south_two.sort_values("log_reviews").iloc[0]   # lower demand in south
south_high = south_two.sort_values("log_reviews").iloc[1]   # higher demand in south
north_low  = north_two.sort_values("log_reviews").iloc[0]
north_high = north_two.sort_values("log_reviews").iloc[1]
middle     = middle_one.iloc[0]

# Canonical labels wanted:
# 0=South low (blue), 1=North high (purple), 2=North low (pink),
# 3=Middle/rural (orange), 4=South high (yellow)
label_map = {
    south_low["orig_label"]: 0,
    north_high["orig_label"]: 1,
    north_low["orig_label"]: 2,
    middle["orig_label"]: 3,
    south_high["orig_label"]: 4,
}

# Predict on FILTERED subset (no retrain) and REMAP to canonical labels
if df_f.empty:
    df_f = df.copy().head(0)
    df_f["cluster"] = []
    df_f["cluster_str"] = []
else:
    X_f = df_f[["latitude","longitude","avg_rating","log_reviews","price_ordinal"]].copy()
    X_f_scaled = scaler.transform(X_f)
    preds = kmeans.predict(X_f_scaled)                  # original kmeans labels
    preds_canonical = pd.Series(preds).map(label_map).astype(int)
    df_f = df_f.copy()
    df_f["cluster"] = preds_canonical
    df_f["cluster_str"] = df_f["cluster"].astype(str)

# Fixed colors consistent with your reference
# 0 blue (south), 1 purple (north high), 2 pink (north lower),
# 3 orange (rural/middle), 4 yellow (south high)
cluster_colors = {
    "0": "#240691",  # blue
    "1": "#7b05a6",  # purple
    "2": "#cc4778",  # pink
    "3": "#f8963f",  # orange
    "4": "#f6e523",  # yellow
}



# =============================
# 1) Clusters map
# =============================
st.markdown("---")
st.markdown("## 1) Cluster Map (K=5)")
if df_f.empty:
    st.warning("No data for clusters map.")
else:
    fig_clusters = px.scatter_mapbox(
        df_f,
        lat="latitude",
        lon="longitude",
        color="cluster_str",
        hover_name="name",
        hover_data={"avg_rating": True, "num_of_reviews": True, "price": True, "state": True},
        zoom=5.2,
        height=650,
        color_discrete_map=cluster_colors,
        category_orders={"cluster_str": ["0","1","2","3","4"]},
    )
    fig_clusters.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Cluster"
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

# =============================
# 2) Cluster profile + interpretation
# =============================
st.markdown("---")
st.markdown("## 2) Cluster profile (K=5) + Suggested interpretation")

def mode_or_nan(s):
    vc = s.value_counts()
    return vc.index[0] if len(vc) else np.nan

if df_f.empty:
    st.info("No data for clusters.")
else:
    profile = (
        df_f.groupby("cluster").agg(
            n=("name","count"),
            rating_mean=("avg_rating","mean"),
            reviews_median=("num_of_reviews","median"),
            price_mode=("price", mode_or_nan),
            open_pct=("state", lambda x: (x=="open").mean()),
            closed_temp_pct=("state", lambda x: (x=="closed_temp").mean()),
            perm_closed_pct=("state", lambda x: (x=="permanently_closed").mean()),
        ).round(2).sort_values("n", ascending=False)
    )
    st.dataframe(profile, use_container_width=True)

    with st.expander("Clusters interpretation"):
        st.markdown("""
- **c0** (blue): Low-price bakeries with high ratings and moderate demand (Southern California; oversupplied/competitive).
- **c1** (purple): Medium-price bakeries with very high popularity (San Francisco Bay Area; highly competitive).
- **c2** (pink): Low-price bakeries with good ratings but fewer reviews (Bay Area / Central Valley; fragmented).
- **c3** (orange): Low-rated, low-demand areas (rural/intermediate zones; higher risk).
- **c4** (yellow): Medium-price bakeries with high demand (Los Angeles / Orange County; dense and competitive).
        """)

# =============================
# 3) Prices and ratings
# =============================
st.markdown("---")
st.markdown("## 3) Prices and ratings")
if df_f.empty:
    st.info("No data to display.")
else:
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Ratings by price level (boxplot)**")
        fig_box = px.box(
            df_f, x="price", y="avg_rating",
            category_orders={"price": ["low","medium","high","unknown"]},
            labels={"price":"Price level","avg_rating":"Average rating"},
            height=360, points="suspectedoutliers"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with cB:
        st.markdown("**Number of businesses by price level**")
        counts_price = (
            df_f["price"]
            .value_counts()
            .reindex(["low","medium","high","unknown"])
            .fillna(0)
            .rename_axis("price").reset_index(name="count")
        )
        fig_bar = px.bar(
            counts_price, x="price", y="count",
            labels={"price":"Price level","count":"# of businesses"},
            height=360
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        st.markdown("**Distribution of ratings (histogram)**")
        fig_hist = px.histogram(
            df_f, x="avg_rating", nbins=25, height=360,
            labels={"avg_rating":"Average rating"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with cD:
        st.markdown("**Reviews vs rating (color = price)**")
        fig_sc = px.scatter(
            df_f, x="num_of_reviews", y="avg_rating",
            color="price",
            category_orders={"price": ["low","medium","high","unknown"]},
            labels={"num_of_reviews":"# of reviews","avg_rating":"Rating"},
            height=360, opacity=0.6
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# =============================
# 4) Map by operational status
# =============================
st.markdown("---")
st.markdown("## 4) Map by operational status")
if df_f.empty:
    st.info("No data for the operational status map.")
else:
    state_colors = {
        "open":"#2ca02c",
        "closed_temp":"#ff7f0e",
        "permanently_closed":"#d62728",
        "unknown":"#7f7f7f"
    }
    df_state = df_f.copy()
    df_state["state"] = df_state["state"].fillna("unknown")
    fig_state = px.scatter_mapbox(
        df_state, lat="latitude", lon="longitude",
        color="state",
        color_discrete_map=state_colors,
        hover_name="name",
        hover_data={"avg_rating":True,"num_of_reviews":True,"price":True,"state":True},
        zoom=5.2, height=620
    )
    fig_state.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_state, use_container_width=True)

# =============================
# 5) Dataset preview
# =============================
st.markdown("## 5) Dataset")
st.dataframe(df_f.head(100), use_container_width=True)
st.caption(f"Showing 100 of {len(df_f):,} records.")
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download (CSV)", data=csv, file_name="bakeries_filtered.csv", mime="text/csv")
