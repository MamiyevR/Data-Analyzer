# FinCatch Interview Project

## Installation Guide

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/FinCatch.git
   cd FinCatch
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

Due to time limitations, I couldn't complete 3rd task. However, here's how I would have implemented it:

<ul>
<li>**Data Preparation**: Extract embeddings from the structured dataset (Q1) and causal graph (Q2), combining node embeddings and relationship data from Neo4j.</li>
<li>**Clustering Algorithm**:
We can ese graph-based clustering (e.g., Neo4j's Louvain or Label Propagation) to group nodes based on relationships.
Alternatively, we can apply embedding-based clustering (e.g., K-Means or DBSCAN) on the vectorized data to group similar entities based on semantic content.</li>
<li>**Evaluation**: Measure clustering quality using metrics like Modularity (for graph-based clusters) or Silhouette Score (for embedding-based clusters).</li>
<li>
**Visualization**:
For graph-based clusters, use Neo4j Bloom or Python's networkx to visualize clusters and their relationships.
For embedding-based clusters, use t-SNE/UMAP with matplotlib to project and display clusters in 2D.M</li>
</ul>
