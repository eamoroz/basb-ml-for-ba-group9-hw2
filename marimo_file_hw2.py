import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import pairwise_distances
    return KMeans, PCA, StandardScaler, pd, plt, silhouette_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Downloading and preparing data
    """)
    return


@app.cell
def _(mo, pd):
    csv_path = mo.notebook_dir() / "insurance_dataset.csv"
    df = pd.read_csv(str(csv_path))
    df.tail()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Describing Data
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    # Filling NaNs
    df["medical_history"] = df["medical_history"].fillna("Unknown")
    df["family_medical_history"] = df["family_medical_history"].fillna("Unknown")
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df, pd):
    pd.crosstab(df["medical_history"], df["family_medical_history"], normalize="index")
    return


@app.cell
def _(df, pd):
    pd.crosstab(df["occupation"], df["region"], normalize="index")
    return


@app.cell
def _(df, pd):
    pd.crosstab(df["gender"], df["exercise_frequency"], normalize="index")
    return


@app.cell
def _(df, plt):
    corr_matrix = df[["age", "bmi", "children", "charges"]].corr()

    plt.figure(figsize=(7, 6))
    plt.imshow(corr_matrix, cmap='coolwarm',interpolation='none')
    plt.colorbar()
    plt.title("Correlation Heatmap")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model 0: Baseline clustering model

    Here we perform a simple baseline segmentation using all available features except coverage_level and charges (they can not be considered as client's characteristics, so they should not be included in the clustering model as features).

    Categorical variables are transformed via one-hot encoding, and all features are standardized with StandardScaler.

    We run K-Means across multiple values of k (1–19) and analyze the Elbow Method to understand how sum of squared errors decreases as the number of clusters grows.
    """)
    return


@app.cell
def _(KMeans, StandardScaler, df, pd, plt):
    cols = ['coverage_level', 'charges']
    df_clust = df.drop(columns=cols)
    df_removed = df[cols].copy()
    X_encoded = pd.get_dummies(df_clust, drop_first=True)
    final_df = pd.concat([X_encoded, df_removed], axis=1)




    cols = ['coverage_level', 'charges']
    df_clust = df.drop(columns=cols)
    X_encoded = pd.get_dummies(df_clust, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X_encoded.index)
    _iter1 = []
    _K = range(1, 20)
    for _k in _K:
        _kmeans = KMeans(n_clusters=_k, random_state=42)
        _kmeans.fit(X_scaled)
        _iter1.append(_kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(_K, _iter1, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors')
    plt.title('Elbow Method for Optimal k')
    plt.grid()
    plt.show()
    return X_encoded, X_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although the sum of squared erorrs steadily decreases with increasing k, there is no clear "elbow point", which makes it difficult to select the optimal number of clusters solely based on this plot.

    This suggests that the data may not have strongly separated natural groups, and additional techniques (e.g., silhouette score or PCA) may be needed to support cluster selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fisrtly, we add the PCA to evaluate whether dimensionality reduction helps reveal clearer cluster structure.
    """)
    return


@app.cell
def _(KMeans, PCA, X_scaled, plt):
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_1 = PCA(n_components=10, random_state=42)
    print('Explained variance ratio sum:', round(sum(pca.explained_variance_ratio_), 3))
    _iter1 = []
    _K = range(1, 20)
    for _k in _K:
        _kmeans = KMeans(n_clusters=_k, random_state=42)
        _kmeans.fit(X_pca)
        _iter1.append(_kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(_K, _iter1, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors')
    plt.title('Elbow Method for Optimal k')
    plt.grid()
    plt.show()
    return (X_pca,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In addition to the Elbow method, we use the Silhouette Score to select the optimal number of clusters (k). The Silhouette Score measures how well each data point fits within its assigned cluster compared to other clusters, capturing both cluster cohesion and separation.
    """)
    return


@app.cell
def _(KMeans, X_pca, plt, silhouette_score):
    # Silhouette_score on data with PCA
    _scores = {}
    _K = range(2, 20)
    for _k in _K:
        _kmeans = KMeans(n_clusters=_k, random_state=42)
        _labels = _kmeans.fit_predict(X_pca)
        _score = silhouette_score(X_pca, _labels, sample_size=10000, random_state=42)
        _scores[_k] = _score
        print(f'k={_k}, silhouette={_score:.4f}')
    plt.plot(list(_scores.keys()), list(_scores.values()), marker='o')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score (with PCA)')
    plt.grid()
    plt.show()
    return


@app.cell
def _(KMeans, X_scaled, plt, silhouette_score):
    # Silhouette_score on data without PCA
    _scores = {}
    _K = range(2, 20)
    for _k in _K:
        print(_k)
        _kmeans = KMeans(n_clusters=_k, random_state=42)
        _labels = _kmeans.fit_predict(X_scaled)
        _score = silhouette_score(X_scaled, _labels, sample_size=10000, random_state=42)
        _scores[_k] = _score
        print(f'k={_k}, silhouette={_score:.4f}')
    plt.plot(list(_scores.keys()), list(_scores.values()), marker='o')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score (without PCA)')
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can see that the most optimal k is in the range of 3–5 using the silhouette score or the Elbow method, so we will present the results using this number of clusters. It is also important for an insurance company to keep the number of customer segments limited, because each distinct policy type with specific conditions creates additional operational and administrative costs. Therefore, choosing a smaller number of clusters helps balance analytical accuracy with business efficiency.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Function for sliders (for choosing the parametrs of clustering and excluding features)
    """)
    return


@app.cell
def _(X_encoded):
    X_encoded.columns
    return


@app.cell
def _(mo):
    slider_children = mo.ui.slider(
      start=-1,
      stop=1,
      value=0,
      step=1,
      label='Client has children. Choose 1 if yes, -1 otherwise.',
      show_value=True
    )

    slider_sportsman = mo.ui.slider(
      start=-1,
      stop=1,
      value=0,
      step=1,
      label='Client does sports frequently. Choose 1 if yes, -1 otherwise.',
      show_value=True
    )
    slider_smoker = mo.ui.slider(
      start=-1,
      stop=1,
      value=0,
      step=1,
      label='Client smokes. Choose 1 if yes, -1 otherwise.',
      show_value=True
    )

    slider_dropgender = mo.ui.slider(
      start=0,
      stop=1,
      value=0,
      step=1,
      label='Drop gender to avoid potential discrimination? Choose 1 if yes, 0 otherwise.',
      show_value=True
    )
    return slider_children, slider_dropgender, slider_smoker, slider_sportsman


@app.function
def filter_data(X_encoded, has_children=None, sportsman = None, only_smokers=None, drop_gender = None):
    data = X_encoded.copy()

    if drop_gender == 1:
        data = data.drop(columns=["gender_male"])

    if has_children == 1:
        data = data[data["children"] > 0]
    elif has_children == -1:
        data = data[data["children"] == 0]

    if sportsman == 1:
        data = data[(data["exercise_frequency_Rarely"] == False) & (data["exercise_frequency_Never"] == False) & (data["exercise_frequency_Occasionally"] == False)]
    elif sportsman == -1:
        data = data[(data["exercise_frequency_Rarely"] ==  True) | (data["exercise_frequency_Never"] == True) | (data["exercise_frequency_Occasionally"] == True)]

    if only_smokers == 1:
        data = data[data["smoker_yes"] == True]
    elif only_smokers is -1:
        data = data[data["smoker_yes"] == False]
    print(f"We are making clustering on dataset with {len(data)} rows and {len(data.columns)} columns.")

    return data


@app.cell
def _(slider_children):
    slider_children
    return


@app.cell
def _(slider_dropgender):
    slider_dropgender
    return


@app.cell
def _(slider_sportsman):
    slider_sportsman
    return


@app.cell
def _(slider_smoker):
    slider_smoker
    return


@app.cell
def _(
    X_encoded,
    slider_children,
    slider_dropgender,
    slider_smoker,
    slider_sportsman,
):
    data_check_1 = filter_data(X_encoded, has_children=slider_children.value, sportsman = slider_sportsman.value, only_smokers=slider_smoker.value, drop_gender = slider_dropgender.value)
    return (data_check_1,)


@app.cell
def _(data_check_1):
    data_check_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Function for clustering
    """)
    return


@app.cell
def _(mo):
    slider_clusters = mo.ui.slider(
      start=1,
      stop=20,
      value=1,
      step=1,
      label='Select the number of clusters:',
      show_value=True)
    return (slider_clusters,)


@app.cell
def _(KMeans, PCA, StandardScaler, plt):
    def cluster_segment(df_filtered, k=3, use_pca_for_model=False):

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_filtered)

        pca_model = PCA(n_components=10, random_state=42)
        X_pca_10 = pca_model.fit_transform(X_scaled)

        X_for_kmeans = X_pca_10 if use_pca_for_model else X_scaled

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_for_kmeans)

        pca_vis = PCA(n_components=2, random_state=42)
        X_pca_2 = pca_vis.fit_transform(X_scaled)

        df_viz = df_filtered.copy()
        df_viz["cluster"] = labels
        df_viz["pca_1"] = X_pca_2[:, 0]
        df_viz["pca_2"] = X_pca_2[:, 1]

        fig = plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            df_viz["pca_1"],
            df_viz["pca_2"],
            c=df_viz["cluster"],
            cmap="viridis"
        )
        plt.title(f"PCA scatter of clusters (k={k})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label="Cluster")

        return fig
    return (cluster_segment,)


@app.cell
def _(slider_clusters):
    slider_clusters
    return


@app.cell
def _(X_encoded, cluster_segment, slider_clusters):
    cluster_segment(X_encoded,k = slider_clusters.value)
    return


@app.cell
def _(KMeans, PCA, StandardScaler):
    def cluster_segment_data(df_filtered, k=3, use_pca_for_model=False):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_filtered)

        pca_model = PCA(n_components=10, random_state=42)
        X_pca_10 = pca_model.fit_transform(X_scaled)

        X_for_kmeans = X_pca_10 if use_pca_for_model else X_scaled

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_for_kmeans)

        pca_vis = PCA(n_components=2, random_state=42)
        X_pca_2 = pca_vis.fit_transform(X_scaled)

        df_viz = df_filtered.copy()
        df_viz["cluster"] = labels
        df_viz["pca_1"] = X_pca_2[:, 0]
        df_viz["pca_2"] = X_pca_2[:, 1]


        return df_viz
    return (cluster_segment_data,)


@app.cell
def _(X_encoded, cluster_segment_data, mo, slider_clusters):
    df_viz = cluster_segment_data(X_encoded,k = slider_clusters.value)

    cluster_profile = df_viz.groupby("cluster").mean()

    mo.ui.table(cluster_profile.round(3))
    return


if __name__ == "__main__":
    app.run()
