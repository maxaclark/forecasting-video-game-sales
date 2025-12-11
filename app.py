import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------- Config ----------
st.set_page_config(
    page_title="Video Game Sales – Prototype Dashboard",
    page_icon="🎮",
    layout="wide",
)

DATA_PATH = Path("data/vgsales_clean.csv")  # fixed dataset


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned video game sales dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def main():
    st.title("🎮 Forecasting Global Video Game Sales – Prototype Dashboard")

    # ---------- Load dataset ----------
    if not DATA_PATH.exists():
        st.error("`data/vgsales_clean.csv` not found. "
                 "Make sure you run this from the repo root and that the file exists.")
        st.stop()

    try:
        df = load_data()
    except Exception as e:
        st.error("Failed to load dataset.")
        st.code(str(e))
        st.stop()

    st.caption(f"Using dataset: `{DATA_PATH}`")
    

    # Basic sanity checks
    required_cols = {"Year", "Genre", "Platform", "Global_Sales"}
    missing = required_cols - set(df.columns)
    if missing:
        st.warning(f"Missing expected columns: {missing}")
        st.stop()

    # Sidebar navigation
    page = st.sidebar.radio(
        "Go to",
        ["Sales Trends", "Prediction (stub)", "Feature Insights (stub)"],
        index=0,
    )

    if page == "Sales Trends":
        render_sales_trends(df)
    elif page == "Prediction (Prototype)":
        render_prediction(df)
    else:
        render_feature_insights_stub()


# ---------- Page 1: Sales Trends ----------

def render_sales_trends(df: pd.DataFrame):
    st.write(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader(" Sales Trend Visualizer")

    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    col_filters, col_plot = st.columns([1, 3])

    with col_filters:
        # Year range
        years = sorted(df["Year"].unique())
        year_min, year_max = int(min(years)), int(max(years))

        year_range = st.slider(
            "Release year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1,
        )

        # Genre filter
        genres = sorted(df["Genre"].dropna().unique())
        selected_genres = st.multiselect(
            "Genre(s)",
            options=genres,
            default=genres,
        )

        # Platform filter
        platforms = sorted(df["Platform"].dropna().unique())
        selected_platforms = st.multiselect(
            "Platform(s)",
            options=platforms,
            default=platforms,
        )

        # Aggregation level
        agg_level = st.radio(
            "Aggregate by",
            ["Year", "Genre", "Platform"],
            index=0,
        )

    # Apply filters
    mask = (
        df["Year"].between(year_range[0], year_range[1])
        & df["Genre"].isin(selected_genres)
        & df["Platform"].isin(selected_platforms)
    )
    df_filtered = df[mask]

    if df_filtered.empty:
        col_plot.warning("No data for this combination of filters.")
        return

    # Group and plot
    if agg_level == "Year":
        grouped = df_filtered.groupby("Year", as_index=False)["Global_Sales"].sum()
        x_col = "Year"
    elif agg_level == "Genre":
        grouped = df_filtered.groupby("Genre", as_index=False)["Global_Sales"].sum()
        x_col = "Genre"
    else:  # Platform
        grouped = df_filtered.groupby("Platform", as_index=False)["Global_Sales"].sum()
        x_col = "Platform"

    grouped = grouped.rename(columns={"Global_Sales": "Total_Global_Sales"})

    if agg_level == "Year":
        fig = px.line(
            grouped,
            x=x_col,
            y="Total_Global_Sales",
            markers=True,
            title="Total Global Sales by Year",
        )
    else:
        fig = px.bar(
            grouped,
            x=x_col,
            y="Total_Global_Sales",
            title=f"Total Global Sales by {agg_level}",
        )

    with col_plot:
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top games under current filters")
        cols_to_show = ["Name", "Platform", "Year", "Genre", "Global_Sales"]
        cols_to_show = [c for c in cols_to_show if c in df_filtered.columns]
        top_games = (
            df_filtered.sort_values("Global_Sales", ascending=False)
            .loc[:, cols_to_show]
            .head(15)
        )
        st.dataframe(top_games, use_container_width=True)


# ---------- Page 2: Prediction (stub) ----------
def render_prediction(df: pd.DataFrame):
    st.subheader("What-If Sales Prediction (Prototype)")

    st.markdown("""
    This page provides a **prototype interface** for exploring hypothetical game releases.
    Users can specify metadata such as title, genre, release year, and platform to simulate 
    how these factors relate to sales performance.

    **Note:**  
    Integrating the trained regression model into the dashboard requires the full preprocessing 
    pipeline (categorical encoders, numeric scalers, and BERT-based title embeddings).  
    As described in the final report, this end-to-end pipeline is not included in the current 
    prototype version and will be incorporated in future work.
    """)

    years = sorted(df["Year"].dropna().astype(int).unique())
    genres = sorted(df["Genre"].dropna().unique())
    platforms = sorted(df["Platform"].dropna().unique())

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Game Title", value="New Awesome Game")
            genre = st.selectbox("Genre", options=genres)
            publisher = st.text_input("Publisher", value="Unknown")

        with col2:
            year = st.selectbox("Release Year", options=years, index=len(years)-1)
            platform = st.selectbox("Platform", options=platforms)

        submitted = st.form_submit_button("Simulate")

    if submitted:
        st.info("""
            ### Prediction Pipeline Pending Integration  
            The trained models (Neural Network, XGBoost, LightGBM) rely on:
            - Numerical scaling  
            - One-hot encoding for categorical features  
            - 768-dimensional **BERT embeddings** for game titles  
            - A specific feature ordering used during training  

            These components are part of the research workflow described in the final report 
            and will be integrated into the dashboard in a future iteration.
        """)

        st.json({
            "Title": title,
            "Year": int(year),
            "Genre": genre,
            "Publisher": publisher,
            "Platform": platform
        })


# ---------- Page 3: Feature Insights (stub) ----------

def render_feature_insights_stub():
    st.subheader("Feature Insights (SHAP & Correlations – Planned)")

    st.markdown(
        """
        This page is planned to show:
        - **SHAP summary plots** for the best regression model  
        - **Correlation heatmap** of numeric / engineered features  
        - Possibly **slice analysis** by Genre / Platform / Year buckets  

        For the final report, PNGs from Jupyter notebooks can be exported
        and displayed here with `st.image(...)`.
        """
    )


if __name__ == "__main__":
    main()
