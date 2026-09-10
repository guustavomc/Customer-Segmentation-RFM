"""
Segmentação de Clientes com Aprendizado Não Supervisionado
=============================================================

Pipeline completo: carrega o UCI Online Retail Dataset, calcula RFM
(Recency, Frequency, Monetary), roda e compara K-Means / Hierarchical /
DBSCAN, gera as visualizações do Item 2 e exporta tudo (CSVs, PNGs,
HTMLs interativos) para a pasta `outputs/`.

Uso:
    python segmentacao_clientes.py
    python segmentacao_clientes.py --data "Online Retail.xlsx"

Requisitos:
    pip install pandas numpy scikit-learn scipy seaborn matplotlib plotly openpyxl ucimlrepo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
FEATURES = ["Recency", "Frequency", "Monetary"]
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)


# ---------------------------------------------------------------------------
# 1. Carregamento dos dados
# ---------------------------------------------------------------------------
def carregar_dados(caminho_local: str | None = None) -> pd.DataFrame:
    """Carrega o UCI Online Retail Dataset.

    Ordem de tentativa: (a) arquivo local informado via --data,
    (b) pacote ucimlrepo, (c) download direto do .xlsx oficial.
    """
    if caminho_local:
        path = Path(caminho_local)
        df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(
            path, encoding="ISO-8859-1"
        )
        print(f"Dados carregados de {path}")
        return df

    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=352)
        print("Dados carregados via ucimlrepo.")
        return dataset.data.original
    except Exception as e:
        print(f"ucimlrepo falhou ({e}), tentando download direto...")

    try:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00352/Online%20Retail.xlsx"
        )
        df = pd.read_excel(url)
        print("Dados carregados via download direto.")
        return df
    except Exception as e:
        raise FileNotFoundError(
            "Não foi possível carregar os dados automaticamente "
            f"({e}). Baixe manualmente em "
            "https://archive.ics.uci.edu/dataset/352/online+retail e "
            "rode novamente com --data 'Online Retail.xlsx'."
        ) from e


# ---------------------------------------------------------------------------
# 2. Limpeza
# ---------------------------------------------------------------------------
def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.drop_duplicates()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    print(f"Linhas após limpeza: {len(df):,} | Clientes únicos: {df['CustomerID'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# 3. RFM
# ---------------------------------------------------------------------------
def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    return rfm


def preparar_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, RobustScaler]:
    q_low, q_high = rfm["Monetary"].quantile([0.01, 0.99])
    rfm_model = rfm[(rfm["Monetary"] >= q_low) & (rfm["Monetary"] <= q_high)].copy()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(rfm_model[FEATURES])
    print(f"Clientes na modelagem: {len(rfm_model):,} (de {len(rfm):,} originais)")
    return rfm_model, X_scaled, scaler


# ---------------------------------------------------------------------------
# 4. Clustering — K-Means / Hierarchical / DBSCAN
# ---------------------------------------------------------------------------
def rodar_kmeans(X_scaled: np.ndarray, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inércia (WCSS)")
    axes[1].plot(list(k_range), silhouettes, marker="o", color="darkorange")
    axes[1].set_title("Silhouette Score por k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "kmeans_elbow_silhouette.png", dpi=150)
    plt.close(fig)

    k_otimo = list(k_range)[int(np.argmax(silhouettes))]
    print(f"[K-Means] k sugerido pelo silhouette: {k_otimo} (ajuste manualmente se o Elbow discordar)")

    t0 = time.time()
    modelo = KMeans(n_clusters=k_otimo, random_state=RANDOM_STATE, n_init=10)
    labels = modelo.fit_predict(X_scaled)
    tempo = time.time() - t0
    return labels, k_otimo, tempo


def rodar_hierarchical(X_scaled: np.ndarray, k_otimo: int):
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_scaled), size=min(500, len(X_scaled)), replace=False
    )
    Z = linkage(X_scaled[sample_idx], method="ward")
    fig = plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=30)
    plt.title("Dendrograma (Ward, amostra de clientes)")
    plt.xlabel("Clientes (agrupados)")
    plt.ylabel("Distância")
    fig.savefig(FIG_DIR / "hierarchical_dendrogram.png", dpi=150)
    plt.close(fig)

    scores = {}
    for method in ["ward", "complete", "average"]:
        labels = AgglomerativeClustering(n_clusters=k_otimo, linkage=method).fit_predict(X_scaled)
        scores[method] = silhouette_score(X_scaled, labels)
        print(f"[Hierarchical] linkage={method:10s} silhouette={scores[method]:.4f}")

    melhor = max(scores, key=scores.get)
    t0 = time.time()
    labels = AgglomerativeClustering(n_clusters=k_otimo, linkage=melhor).fit_predict(X_scaled)
    tempo = time.time() - t0
    print(f"[Hierarchical] melhor linkage: {melhor}")
    return labels, melhor, tempo


def rodar_dbscan(X_scaled: np.ndarray, eps: float | None = None):
    min_samples = X_scaled.shape[1] * 2
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    fig = plt.figure(figsize=(8, 4))
    plt.plot(k_distances)
    plt.title(f"K-distance graph (k={min_samples})")
    plt.xlabel("Pontos ordenados")
    plt.ylabel(f"Distância ao {min_samples}º vizinho")
    fig.savefig(FIG_DIR / "dbscan_k_distance.png", dpi=150)
    plt.close(fig)

    # heurística simples de cotovelo caso eps não seja informado manualmente
    if eps is None:
        deltas = np.diff(k_distances)
        eps = float(k_distances[max(int(np.argmax(deltas)), 1)])
        print(f"[DBSCAN] eps estimado automaticamente pelo cotovelo: {eps:.3f} "
              f"(revise o gráfico dbscan_k_distance.png e ajuste com --eps se necessário)")

    t0 = time.time()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    tempo = time.time() - t0
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = list(labels).count(-1)
    print(f"[DBSCAN] {n_clusters} clusters | {n_ruido} pontos de ruído ({n_ruido/len(labels):.1%})")
    return labels, eps, tempo


def comparar_algoritmos(X_scaled, resultados: dict[str, tuple[np.ndarray, float]]) -> pd.DataFrame:
    linhas = []
    for nome, (labels, tempo) in resultados.items():
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        if n_clusters < 2:
            linhas.append({"algoritmo": nome, "n_clusters": n_clusters, "silhouette": np.nan,
                            "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
                            "tempo_execucao_s": tempo})
            continue
        linhas.append({
            "algoritmo": nome,
            "n_clusters": n_clusters,
            "silhouette": silhouette_score(X_scaled[mask], labels[mask]),
            "davies_bouldin": davies_bouldin_score(X_scaled[mask], labels[mask]),
            "calinski_harabasz": calinski_harabasz_score(X_scaled[mask], labels[mask]),
            "tempo_execucao_s": tempo,
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 5. Visualizações (Item 2)
# ---------------------------------------------------------------------------
def visualizar_pca(rfm_model: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    coords2 = pca2.fit_transform(X_scaled)
    rfm_model["PC1"], rfm_model["PC2"] = coords2[:, 0], coords2[:, 1]
    print(f"[PCA 2D] variância explicada: {pca2.explained_variance_ratio_.sum():.1%}")

    fig = px.scatter(rfm_model, x="PC1", y="PC2", color=rfm_model["Cluster"].astype(str),
                      title="Clusters de Clientes — Projeção PCA 2D", opacity=0.7,
                      labels={"color": "Cluster"})
    fig.write_html(FIG_DIR / "pca_2d.html")

    pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
    coords3 = pca3.fit_transform(X_scaled)
    rfm_model["PC3"] = coords3[:, 2]
    print(f"[PCA 3D] variância explicada: {pca3.explained_variance_ratio_.sum():.1%}")

    fig3 = px.scatter_3d(rfm_model, x="PC1", y="PC2", z="PC3", color=rfm_model["Cluster"].astype(str),
                          title="Clusters de Clientes — Projeção PCA 3D", opacity=0.7,
                          labels={"color": "Cluster"})
    fig3.write_html(FIG_DIR / "pca_3d.html")
    return rfm_model


def radar_chart(cluster_means: pd.DataFrame):
    norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    fig = go.Figure()
    for cluster in norm.index:
        fig.add_trace(go.Scatterpolar(r=norm.loc[cluster].values, theta=FEATURES,
                                       fill="toself", name=f"Cluster {cluster}"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       title="Perfil dos Clusters (RFM normalizado)", showlegend=True)
    fig.write_html(FIG_DIR / "radar_clusters.html")
    return norm


def heatmap_clusters(cluster_means: pd.DataFrame, norm: pd.DataFrame):
    fig = plt.figure(figsize=(8, 5))
    sns.heatmap(norm, annot=cluster_means.round(1), fmt="g", cmap="YlOrRd",
                cbar_kws={"label": "Valor normalizado"})
    plt.title("Heatmap de Características por Cluster")
    plt.ylabel("Cluster")
    fig.savefig(FIG_DIR / "heatmap_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def distribuicao_por_segmento(rfm_model: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    rfm_model["Cluster"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Nº de Clientes por Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Clientes")

    sns.boxplot(data=rfm_model, x="Cluster", y="Monetary", ax=axes[1])
    axes[1].set_title("Distribuição de Valor (Monetary) por Cluster")

    revenue = rfm_model.groupby("Cluster")["Monetary"].sum()
    axes[2].pie(revenue, labels=[f"Cluster {c}" for c in revenue.index], autopct="%1.1f%%", startangle=90)
    axes[2].set_title("Contribuição de Revenue por Segmento")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "distribuicao_segmentos.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Nomeação dos clusters (framework RFM)
# ---------------------------------------------------------------------------
def nomear_cluster(row: pd.Series, medianas: pd.Series) -> str:
    r_baixo = row["Recency"] <= medianas["Recency"]
    f_alto = row["Frequency"] >= medianas["Frequency"]
    m_alto = row["Monetary"] >= medianas["Monetary"]

    if r_baixo and f_alto and m_alto:
        return "Champions"
    if r_baixo and f_alto:
        return "Loyal Customers"
    if r_baixo and not f_alto:
        return "New Customers"
    if not r_baixo and (f_alto or m_alto):
        return "At Risk"
    return "Lost"


def montar_perfil_clusters(rfm_model: pd.DataFrame) -> pd.DataFrame:
    cluster_means = rfm_model.groupby("Cluster")[FEATURES].mean()
    medianas = rfm_model[FEATURES].median()

    perfil = cluster_means.copy()
    perfil["Nome_Sugerido"] = perfil.apply(lambda row: nomear_cluster(row, medianas), axis=1)
    perfil["N_Clientes"] = rfm_model["Cluster"].value_counts().sort_index()
    perfil["Revenue_Total"] = rfm_model.groupby("Cluster")["Monetary"].sum()
    perfil["Revenue_%"] = (perfil["Revenue_Total"] / perfil["Revenue_Total"].sum() * 100).round(1)
    return perfil.sort_values("Revenue_Total", ascending=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None,
                         help="Caminho para um Online Retail.xlsx/.csv já baixado localmente")
    parser.add_argument("--algoritmo-final", choices=["kmeans", "hierarchical", "dbscan"],
                         default="kmeans", help="Algoritmo usado para os relatórios/visualizações finais")
    parser.add_argument("--eps", type=float, default=None,
                         help="eps do DBSCAN (se omitido, é estimado pelo cotovelo do k-distance graph)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    print("=== 1. Carregando dados ===")
    df = carregar_dados(args.data)

    print("\n=== 2. Limpando dados ===")
    df_clean = limpar_dados(df)

    print("\n=== 3. Calculando RFM ===")
    rfm = calcular_rfm(df_clean)
    rfm_model, X_scaled, _ = preparar_features(rfm)

    print("\n=== 4. Rodando algoritmos de clustering ===")
    labels_kmeans, k_otimo, t_kmeans = rodar_kmeans(X_scaled)
    labels_hier, melhor_linkage, t_hier = rodar_hierarchical(X_scaled, k_otimo)
    labels_dbscan, eps_usado, t_dbscan = rodar_dbscan(X_scaled, eps=args.eps)

    rfm_model["Cluster_KMeans"] = labels_kmeans
    rfm_model["Cluster_Hierarchical"] = labels_hier
    rfm_model["Cluster_DBSCAN"] = labels_dbscan

    comparacao = comparar_algoritmos(X_scaled, {
        "K-Means": (labels_kmeans, t_kmeans),
        "Hierarchical": (labels_hier, t_hier),
        "DBSCAN": (labels_dbscan, t_dbscan),
    })
    comparacao.to_csv(OUTPUT_DIR / "comparacao_algoritmos.csv", index=False)
    print("\n--- Comparação dos algoritmos ---")
    print(comparacao.to_string(index=False))

    coluna_final = {"kmeans": "Cluster_KMeans", "hierarchical": "Cluster_Hierarchical",
                     "dbscan": "Cluster_DBSCAN"}[args.algoritmo_final]
    rfm_model["Cluster"] = rfm_model[coluna_final]
    print(f"\nAlgoritmo final escolhido: {coluna_final}")

    print("\n=== 5. Gerando visualizações ===")
    rfm_model = visualizar_pca(rfm_model, X_scaled)
    cluster_means = rfm_model.groupby("Cluster")[FEATURES].mean()
    norm = radar_chart(cluster_means)
    heatmap_clusters(cluster_means, norm)
    distribuicao_por_segmento(rfm_model)

    print("\n=== 6. Nomeando clusters ===")
    perfil = montar_perfil_clusters(rfm_model)
    perfil.to_csv(OUTPUT_DIR / "perfil_clusters.csv")
    print(perfil.to_string())

    rfm_model.to_csv(OUTPUT_DIR / "clientes_segmentados.csv", index=False)

    print(f"\nConcluído. Resultados em: {OUTPUT_DIR.resolve()}")
    print(f"Figuras (PNG estáticos + HTML interativos) em: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()