"""
Dashboard Interativo — Segmentação de Clientes RFM
====================================================

Lê os CSVs gerados por `main.py` em `outputs/` e monta um dashboard único,
interativo (Plotly), em uma página HTML autocontida — sem precisar de
servidor (Streamlit) para ser visualizado.

Uso:
    python main.py          # gera outputs/*.csv primeiro
    python dashboard.py     # depois monta outputs/dashboard.html
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path("outputs")
FEATURES = ["Recency", "Frequency", "Monetary"]


def carregar_outputs():
    clientes = pd.read_csv(OUTPUT_DIR / "clientes_segmentados.csv")
    perfil = pd.read_csv(OUTPUT_DIR / "perfil_clusters.csv")
    comparacao = pd.read_csv(OUTPUT_DIR / "comparacao_algoritmos.csv")
    return clientes, perfil, comparacao


def algoritmo_final(clientes: pd.DataFrame) -> str:
    mapa = {"K-Means": "Cluster_KMeans", "Hierarchical": "Cluster_Hierarchical", "DBSCAN": "Cluster_DBSCAN"}
    for nome, coluna in mapa.items():
        if coluna in clientes.columns and clientes["Cluster"].equals(clientes[coluna]):
            return nome
    return "K-Means"


def montar_dashboard(clientes: pd.DataFrame, perfil: pd.DataFrame, comparacao: pd.DataFrame) -> go.Figure:
    algo_final = algoritmo_final(clientes)
    metricas_final = comparacao.set_index("algoritmo").loc[algo_final]

    clusters_validos = sorted(c for c in clientes["Cluster"].unique() if c != -1)
    cores = ["#0B7285", "#D9480F", "#5F3DC4", "#2B8A3E", "#B08A2E", "#9C36B5"]
    cor_por_cluster = {c: cores[i % len(cores)] for i, c in enumerate(clusters_validos)}
    cor_por_cluster[-1] = "#B8C4CE"

    fig = make_subplots(
        rows=5,
        cols=4,
        specs=[
            [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
            [{"type": "xy", "colspan": 2}, None, {"type": "polar", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None, {"type": "domain", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None, {"type": "table", "colspan": 2}, None],
        ],
        row_heights=[0.10, 0.24, 0.22, 0.22, 0.22],
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
        subplot_titles=(
            None, None, None, None,
            "Clusters — Projeção PCA 2D", "Perfil dos Clusters (RFM normalizado)",
            "Heatmap de Características por Cluster", "Nº de Clientes por Cluster",
            "Distribuição de Monetary por Cluster", "Contribuição de Revenue por Segmento",
            "Perfil e Nomeação dos Clusters", "Comparação dos Algoritmos",
        ),
    )

    # --- Linha 1: KPIs ---------------------------------------------------
    fig.add_trace(go.Indicator(
        mode="number", value=len(clientes),
        title={"text": "Total de Clientes"}, number={"valueformat": ",.0f"},
    ), row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="number", value=clientes["Monetary"].sum(),
        title={"text": "Receita Total"}, number={"prefix": "£", "valueformat": ",.0f"},
    ), row=1, col=2)
    fig.add_trace(go.Indicator(
        mode="number", value=len(clusters_validos),
        title={"text": "Nº de Segmentos"},
    ), row=1, col=3)
    fig.add_trace(go.Indicator(
        mode="number", value=metricas_final["silhouette"],
        title={"text": f"Silhouette ({algo_final})"}, number={"valueformat": ".3f"},
    ), row=1, col=4)

    # --- Linha 2: PCA 2D + Radar ------------------------------------------
    for c in sorted(clientes["Cluster"].unique()):
        sub = clientes[clientes["Cluster"] == c]
        nome = "Ruído" if c == -1 else int(c)
        fig.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"], mode="markers", name=f"Cluster {nome}",
            marker=dict(color=cor_por_cluster[c], size=5, opacity=0.6),
            legendgroup=f"cluster{c}",
        ), row=2, col=1)

    cluster_means = clientes[clientes["Cluster"] != -1].groupby("Cluster")[FEATURES].mean()
    norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    for c in norm.index:
        fig.add_trace(go.Scatterpolar(
            r=norm.loc[c].values, theta=FEATURES, fill="toself",
            name=f"Cluster {int(c)}", legendgroup=f"cluster{c}", showlegend=False,
            line=dict(color=cor_por_cluster[c]),
        ), row=2, col=3)

    # --- Linha 3: Heatmap + Barras ----------------------------------------
    fig.add_trace(go.Heatmap(
        z=norm.values, x=FEATURES, y=[f"Cluster {int(c)}" for c in norm.index],
        colorscale="YlOrRd", showscale=False,
        text=cluster_means.round(1).values, texttemplate="%{text}",
    ), row=3, col=1)

    contagem = clientes["Cluster"].value_counts().sort_index()
    fig.add_trace(go.Bar(
        x=[f"Cluster {int(c)}" if c != -1 else "Ruído" for c in contagem.index],
        y=contagem.values,
        marker_color=[cor_por_cluster[c] for c in contagem.index],
        showlegend=False,
    ), row=3, col=3)

    # --- Linha 4: Boxplot + Pizza -------------------------------------------
    for c in sorted(clientes["Cluster"].unique()):
        sub = clientes[clientes["Cluster"] == c]
        nome = "Ruído" if c == -1 else f"Cluster {int(c)}"
        fig.add_trace(go.Box(
            y=sub["Monetary"], name=nome, marker_color=cor_por_cluster[c], showlegend=False,
        ), row=4, col=1)

    revenue = clientes[clientes["Cluster"] != -1].groupby("Cluster")["Monetary"].sum()
    fig.add_trace(go.Pie(
        labels=[f"Cluster {int(c)}" for c in revenue.index], values=revenue.values,
        marker=dict(colors=[cor_por_cluster[c] for c in revenue.index]),
        showlegend=False,
    ), row=4, col=3)

    # --- Linha 5: Tabelas ----------------------------------------------------
    perfil_fmt = perfil.copy()
    for col in ["Recency", "Frequency", "Monetary", "Revenue_Total"]:
        perfil_fmt[col] = perfil_fmt[col].round(1)
    fig.add_trace(go.Table(
        header=dict(values=list(perfil_fmt.columns), fill_color="#0B7285", font=dict(color="white"), align="left"),
        cells=dict(values=[perfil_fmt[c] for c in perfil_fmt.columns], align="left"),
    ), row=5, col=1)

    comp_fmt = comparacao.copy()
    for col in comp_fmt.select_dtypes("float").columns:
        comp_fmt[col] = comp_fmt[col].round(4)
    fig.add_trace(go.Table(
        header=dict(values=list(comp_fmt.columns), fill_color="#0B7285", font=dict(color="white"), align="left"),
        cells=dict(values=[comp_fmt[c] for c in comp_fmt.columns], align="left"),
    ), row=5, col=3)

    fig.update_xaxes(title_text="PC1", row=2, col=1)
    fig.update_yaxes(title_text="PC2", row=2, col=1)
    fig.update_yaxes(title_text="Monetary", row=4, col=1)

    fig.update_layout(
        title=dict(
            text=f"Segmentação de Clientes RFM — Dashboard ({algo_final}, k={len(clusters_validos)})",
            x=0.5, font=dict(size=22),
        ),
        height=1500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.27),
        template="plotly_white",
    )
    return fig


def main():
    clientes, perfil, comparacao = carregar_outputs()
    fig = montar_dashboard(clientes, perfil, comparacao)
    out_path = OUTPUT_DIR / "dashboard.html"
    fig.write_html(out_path)
    print(f"Dashboard salvo em: {out_path.resolve()}")


if __name__ == "__main__":
    main()
