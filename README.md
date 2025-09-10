# Amazon-Music

# Amazon Music Clustering

Group similar songs by audio features using unsupervised learning (K-Means), then visualize and interpret clusters.

##  What’s Inside
- `Amazon_Music_Clustering.py` – end-to-end clustering pipeline
- (You provide) `Data_Set.csv` – your songs + audio features
- Outputs:
  - `clustered_music.csv` – original rows + `cluster` label
  - `outputs/` – elbow/silhouette/PCA/t-SNE plots (when `--save-plots` is used)

##  Requirements
```bash
pip install pandas numpy scikit-learn matplotlib
```
(Optional for t-SNE speedups: `pip install openTSNE`)

##  Expected CSV Columns
Audio features (the script will auto-skip missing ones and warn):
```
danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms
```
Optional descriptive columns (kept in export if present):  
`track_id, track_name, artist_name`

##  How to Run

**Basic run:**
```bash
python Amazon_Music_Clustering.py --csv Data_Set.csv --out clustered_music.csv
```

**Save plots (PNG) to `./outputs`:**
```bash
python Amazon_Music_Clustering.py --csv Data_Set.csv --out clustered_music.csv --save-plots
```

**Force a specific k (skip chooser):**
```bash
python Amazon_Music_Clustering.py --csv Data_Set.csv --k 6 --save-plots
```

**Also render t-SNE (slower):**
```bash
python Amazon_Music_Clustering.py --csv Data_Set.csv --save-plots --tsne
```

##  What the Script Does
1. Loads CSV and cleans duplicates.
2. Selects available audio features.
3. Imputes numeric NaNs with median and standardizes features.
4. Runs Elbow Method + Silhouette Sweep to suggest k.
5. Trains K-Means and assigns cluster labels.
6. Builds cluster profiles (feature means, scaled space).
7. Saves plots and the labeled dataset.

##  Interpreting Clusters
- High `energy` + `danceability` → party tracks
- High `acousticness` + low `energy` → chill/acoustic
- High `valence` → happy/positive mood
- High `instrumentalness` → likely instrumentals

##  Metrics
- Silhouette Score (higher is better, max ~1.0)
- Davies-Bouldin Index (lower is better)

##  Tips
- If features are highly skewed (e.g., `duration_ms`), consider log-transforming them first.
- Try DBSCAN or Hierarchical Clustering for non-spherical clusters (you can extend the script).
- For huge datasets, sample first for t-SNE visualization.

---

© 2025 Amazon Music Clustering – for educational use (GUVI submission friendly).

