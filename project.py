import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="🏦",
    layout="wide",
)

sns.set_style("whitegrid")
plt.rcParams["figure.autolayout"] = True

df = pd.read_csv("bank.csv")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

st.sidebar.title("🏦 Bank Marketing EDA")
st.sidebar.markdown("Use the filters below to explore subsets of the data.")

st.sidebar.subheader("Filters")

selected_job = st.sidebar.multiselect(
    "Job", options=sorted(df["job"].unique()), default=[]
)
selected_marital = st.sidebar.multiselect(
    "Marital status", options=sorted(df["marital"].unique()), default=[]
)
selected_education = st.sidebar.multiselect(
    "Education", options=sorted(df["education"].unique()), default=[]
)
selected_deposit = st.sidebar.multiselect(
    "Deposit (target)", options=sorted(df["deposit"].unique()), default=[]
)

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

filtered_df = df.copy()
if selected_job:
    filtered_df = filtered_df[filtered_df["job"].isin(selected_job)]
if selected_marital:
    filtered_df = filtered_df[filtered_df["marital"].isin(selected_marital)]
if selected_education:
    filtered_df = filtered_df[filtered_df["education"].isin(selected_education)]
if selected_deposit:
    filtered_df = filtered_df[filtered_df["deposit"].isin(selected_deposit)]
filtered_df = filtered_df[
    (filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])
]

st.sidebar.markdown(f"**Rows after filtering:** {filtered_df.shape[0]:,} / {df.shape[0]:,}")

st.title("🏦 Bank Marketing Campaign - Exploratory Data Analysis")
st.markdown(
    "A simple, interactive EDA app built with **NumPy**, **Pandas**, "
    "**Matplotlib**, **Seaborn** and **Streamlit**. "
)

tabs = st.tabs(
    [
        "📋 Overview",
        "📊 Univariate",
        "🔗 Relationships",
        "🔥 Correlation",
        "📈 Target Analysis",
    ]
)

with tabs[0]:
    st.subheader("Dataset Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{filtered_df.shape[0]:,}")
    c2.metric("Columns", f"{filtered_df.shape[1]}")
    c3.metric("Numeric features", len(numeric_cols))
    c4.metric("Categorical features", len(categorical_cols))

    st.markdown("**First rows of the (filtered) data:**")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.markdown("**Column data types:**")
    dtype_df = pd.DataFrame(
        {"column": df.columns, "dtype": df.dtypes.astype(str).values}
    )
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("**Missing values:**")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["column", "missing_count"]
    missing_df["missing_%"] = (missing_df["missing_count"] / len(df) * 100).round(2)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.markdown("**Descriptive statistics (numeric columns):**")
    st.dataframe(filtered_df[numeric_cols].describe().T, use_container_width=True)

    st.markdown("**Descriptive statistics (categorical columns):**")
    st.dataframe(
        filtered_df[categorical_cols].describe().T, use_container_width=True
    )

with tabs[1]:
    st.subheader("Univariate Analysis")

    col_type = st.radio(
        "Choose column type", ["Numeric", "Categorical"], horizontal=True
    )

    if col_type == "Numeric":
        col = st.selectbox("Select a numeric column", numeric_cols)

        colA, colB = st.columns(2)

        with colA:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(filtered_df[col], kde=True, ax=ax, color="steelblue")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
            plt.close(fig)

        with colB:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=filtered_df[col], ax=ax, color="salmon")
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Summary statistics:**")
        stats = filtered_df[col].describe()
        skew = filtered_df[col].skew()
        kurt = filtered_df[col].kurt()
        stats_df = stats.to_frame().T
        stats_df["skewness"] = skew
        stats_df["kurtosis"] = kurt
        st.dataframe(stats_df, use_container_width=True)

    else:
        col = st.selectbox("Select a categorical column", categorical_cols)

        counts = filtered_df[col].value_counts()

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x=counts.values, y=counts.index, ax=ax, palette="viridis"
            )
            ax.set_xlabel("Count")
            ax.set_title(f"Frequency of {col}")
            st.pyplot(fig)
            plt.close(fig)

        with colB:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title(f"Share of {col}")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Value counts:**")
        st.dataframe(counts.reset_index(), use_container_width=True, hide_index=True)


with tabs[2]:
    st.subheader("Relationships Between Variables")

    rel_type = st.radio(
        "Relationship type",
        ["Numeric vs Numeric", "Numeric vs Categorical", "Categorical vs Categorical"],
        horizontal=True,
    )

    if rel_type == "Numeric vs Numeric":
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X axis", numeric_cols, index=0)
        y_col = c2.selectbox("Y axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        hue_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)

        fig, ax = plt.subplots(figsize=(8, 5))
        if hue_col == "None":
            sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax, alpha=0.6)
        else:
            sns.scatterplot(
                data=filtered_df, x=x_col, y=y_col, hue=hue_col, ax=ax, alpha=0.6
            )
        ax.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig)
        plt.close(fig)

        corr_val = filtered_df[[x_col, y_col]].corr().iloc[0, 1]
        st.metric("Pearson correlation", f"{corr_val:.3f}")

    elif rel_type == "Numeric vs Categorical":
        c1, c2 = st.columns(2)
        num_col = c1.selectbox("Numeric column", numeric_cols)
        cat_col = c2.selectbox("Categorical column", categorical_cols)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=filtered_df, x=cat_col, y=num_col, ax=ax, palette="Set2")
        ax.set_title(f"{num_col} by {cat_col}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"**{num_col} statistics grouped by {cat_col}:**")
        st.dataframe(
            filtered_df.groupby(cat_col)[num_col].describe(), use_container_width=True
        )

    else:
        c1, c2 = st.columns(2)
        cat_col1 = c1.selectbox("Categorical column 1", categorical_cols, index=0)
        cat_col2 = c2.selectbox(
            "Categorical column 2",
            categorical_cols,
            index=1 if len(categorical_cols) > 1 else 0,
        )

        cross_tab = pd.crosstab(filtered_df[cat_col1], filtered_df[cat_col2])

        fig, ax = plt.subplots(figsize=(9, 5))
        cross_tab.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(f"{cat_col1} vs {cat_col2}")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Cross-tabulation:**")
        st.dataframe(cross_tab, use_container_width=True)

with tabs[3]:
    st.subheader("Correlation Heatmap (numeric columns)")

    corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
    corr_matrix = filtered_df[numeric_cols].corr(method=corr_method)

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
    )
    ax.set_title(f"{corr_method.title()} Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Correlation matrix (table):**")
    st.dataframe(corr_matrix.round(3), use_container_width=True)

with tabs[4]:
    st.subheader("Target Variable Analysis: `deposit`")

    c1, c2 = st.columns(2)

    with c1:
        counts = filtered_df["deposit"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Set1")
        ax.set_ylabel("Count")
        ax.set_title("Deposit Subscription Count")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        rate = (filtered_df["deposit"].value_counts(normalize=True) * 100).round(2)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(rate.values, labels=rate.index, autopct="%1.1f%%", startangle=90,
               colors=sns.color_palette("Set1"))
        ax.set_title("Deposit Subscription Rate")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Deposit rate by categorical feature")
    cat_for_target = st.selectbox(
        "Choose a categorical feature", [c for c in categorical_cols if c != "deposit"]
    )

    rate_by_cat = (
        pd.crosstab(filtered_df[cat_for_target], filtered_df["deposit"], normalize="index")
        * 100
    ).round(2)

    fig, ax = plt.subplots(figsize=(9, 5))
    rate_by_cat.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Deposit Rate by {cat_for_target}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Numeric feature vs deposit outcome")
    num_for_target = st.selectbox("Choose a numeric feature", numeric_cols)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(
        data=filtered_df, x="deposit", y=num_for_target, ax=ax, palette="muted"
    )
    ax.set_title(f"{num_for_target} distribution by deposit outcome")
    st.pyplot(fig)
    plt.close(fig)

