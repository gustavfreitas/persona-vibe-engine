"""Avalia diferentes valores de k com silhouette score e inércia (elbow)."""

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.model import FEATURE_COLUMNS, RANDOM_STATE, clean_data, load_dataset

SAMPLE_SIZE = 10_000  # o silhouette é O(n²): usa uma amostra para ser rápido

df = clean_data(load_dataset(Path("data/dataset.csv")))
scaled = StandardScaler().fit_transform(df[FEATURE_COLUMNS])

print("k | inércia | silhouette")
for k in range(2, 11):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled)
    score = silhouette_score(
        scaled, model.labels_, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE
    )
    print(f"{k} | {model.inertia_:10.0f} | {score:.3f}")