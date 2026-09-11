# Customer Segmentation RFM

> Developed for the Unsupervised Learning course (Aprendizado de Máquina Não Supervisionado) — Specialization in Applied Artificial Intelligence, UNISINOS.

Customer segmentation with unsupervised machine learning, using **RFM** analysis (Recency, Frequency, Monetary) on the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail). The pipeline compares three clustering algorithms (K-Means, Hierarchical and DBSCAN), validates the clusters with quantitative metrics, and produces the visualizations and segment naming required by the assignment (see [arquivos-base/ATIVIDADE.pdf](arquivos-base/ATIVIDADE.pdf)).

## What the pipeline does

1. **Loads the data** - local file (`--data`), via the `ucimlrepo` package, or a direct download of the official `.xlsx` (in that order of attempt).
2. **Cleans the data** - drops rows without `CustomerID`, cancellations (`InvoiceNo` starting with "C"), non-positive quantities/prices, and duplicates.
3. **Computes RFM** per customer (Recency in days, Frequency = number of unique orders, Monetary = total amount spent), removing extreme Monetary outliers (1%-99% percentiles), then applies `log1p` (RFM is heavily right-skewed) before scaling with `RobustScaler`.
4. **Runs and compares the algorithms**:
   - **K-Means** - computes both the Elbow Method (inertia) and Silhouette Score across k=2..10. `k` is chosen from the **Elbow's knee** (kneedle detection) rather than the Silhouette maximum alone, because on RFM data Silhouette alone tends to favor a shallow k=2 split ("high value vs. everyone else") that hides business-relevant segments; the script logs both values and flags when they disagree. Also runs a stability analysis (Adjusted Rand Index across runs with different seeds) and a convergence check (number of iterations).
   - **Hierarchical (Agglomerative)** - dendrogram (sample of up to 500 customers) and comparison across `ward`/`complete`/`average` linkages, picking the one with the highest silhouette.
   - **DBSCAN** - k-distance graph with automatic `eps` estimation via knee detection (kneedle), overridable with `--eps`.
   - Validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, runtime) are saved to `outputs/comparacao_algoritmos.csv`.
5. **Generates the Item 2 visualizations**: PCA 2D/3D scatter plots with marked centroids, a radar chart of cluster profiles, a feature heatmap, and customer/value distribution per segment (bar chart, boxplot, pie chart).
6. **Names the segments** using the classic RFM framework (Champions, Loyal Customers, New Customers, At Risk, Lost), ranking each cluster's Recency/Frequency/Monetary against the other clusters (not against individual customers) so names stay distinct regardless of how many clusters `k` turns out to be. Points DBSCAN flags as noise (when DBSCAN is chosen as the final algorithm) are labeled "Ruido / Outliers" instead of being assigned a segment name.

## Project structure

```
main.py                          # full pipeline (this script)
segmentacao_clientes_rfm.ipynb   # documented, reproducible notebook version of the pipeline
arquivos-base/
  ATIVIDADE.pdf                  # assignment brief
  Codigo Aula 02.ipynb           # example notebooks (synthetic data)
  Codigos Exemplos Modulo 02.ipynb
  Mod2 clustering colabv3.ipynb
  Mod2 Visualizacoes.ipynb
outputs/                         # generated when the script runs (see below)
```

## Requirements

- Python 3.10+
- Dependencies:

```bash
pip install -r requirements.txt
```

## How to run

```bash
# Tries to download the dataset automatically (ucimlrepo, then direct download)
python main.py

# Using a file already downloaded locally (.xlsx or .csv)
python main.py --data "Online Retail.xlsx"

# Choosing which algorithm drives the final visualizations/report
python main.py --algoritmo-final hierarchical

# Fixing DBSCAN's eps manually instead of the auto-estimated value
python main.py --eps 0.75
```

If the automatic download fails, download the dataset manually from https://archive.ics.uci.edu/dataset/352/online+retail and run with `--data`.

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | *(none)* | Path to an `Online Retail.xlsx`/`.csv` already downloaded locally |
| `--algoritmo-final` | `kmeans` | Algorithm used for the final visualizations/profile: `kmeans`, `hierarchical` or `dbscan` |
| `--eps` | *(auto)* | DBSCAN's `eps`; if omitted, it's estimated from the knee of the k-distance graph |

## Generated output (`outputs/`)

| File | Description |
|---|---|
| `comparacao_algoritmos.csv` | Metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, runtime) for the 3 algorithms |
| `perfil_clusters.csv` | Mean RFM, suggested name, customer count and % of revenue per cluster |
| `clientes_segmentados.csv` | Customer base with RFM values and the cluster assigned by each algorithm |
| `figures/kmeans_elbow_silhouette.png` | Elbow Method + Silhouette Score by k |
| `figures/hierarchical_dendrogram.png` | Dendrogram (customer sample) |
| `figures/dbscan_k_distance.png` | K-distance graph used to estimate `eps` |
| `figures/pca_2d.html` / `pca_3d.html` | Interactive PCA scatter with clusters and marked centroids |
| `figures/radar_clusters.html` | Normalized RFM profile per cluster (radar chart) |
| `figures/heatmap_clusters.png` | Heatmap of average feature values per cluster |
| `figures/distribuicao_segmentos.png` | Customer count, Monetary boxplot and revenue contribution per segment |
| `dashboard.html` | Single-page interactive dashboard (generated by `dashboard.py`, see below) |

## Dashboard

```bash
python dashboard.py
```

Reads `outputs/clientes_segmentados.csv`, `perfil_clusters.csv` and `comparacao_algoritmos.csv` (generated by
`main.py`) and assembles a single self-contained interactive page at `outputs/dashboard.html` — KPI cards
(customers, revenue, segments, silhouette of the final model), PCA scatter, radar chart, heatmap, customer/value
distribution, and the cluster profile / algorithm comparison tables. No server required, just open the HTML file
in a browser.

## Notebook

[`segmentacao_clientes_rfm.ipynb`](segmentacao_clientes_rfm.ipynb) walks through the same
pipeline as `main.py`, narrated cell-by-cell (data loading → cleaning → RFM → clustering comparison →
Item 2 visualizations → cluster naming), with all figures and metrics rendered inline. It's self-contained and
reproducible - open it and "Run All" to regenerate everything from scratch.

## Notes

- `RANDOM_STATE = 42` ensures reproducible results.
- The executive report (Item 3 of the assignment) is not included in this repo yet - it's a separate written deliverable, to be built from the numbers and segment names exported here.