# 🎧 Mood Music Recommender

Motor de recomendação musical por *mood*, 100% offline. Agrupa faixas do
[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
(Kaggle) com K-Means, a partir das suas características acústicas, e recomenda
faixas do mesmo cluster através de uma app Streamlit.

## Como funciona

1. **Limpeza:** conversão numérica, remoção de nulos e deduplicação por `track_id`.
2. **Normalização:** `StandardScaler` sobre 7 features (`valence`, `energy`,
   `acousticness`, `instrumentalness`, `danceability`, `loudness`, `tempo`).
3. **Clustering:** `KMeans` com k=5 e `random_state` fixo (reprodutível).
4. **Nomes automáticos:** cada cluster recebe as duas características que mais
   se afastam da média global (ex.: "Suave · Instrumental").
5. **Recomendação:** dentro do cluster (e, opcionalmente, do mesmo género), as
   faixas são ordenadas pela distância euclidiana nas features padronizadas.

## Escolha de k

| k | silhouette |
|---|-----------|
| 2 | 0.311 |
| 3 | 0.201 |
| 4 | 0.208 |
| **5** | **0.218** |
| 6 | 0.199 |
| 10 | 0.175 |

k=5 é o melhor valor a partir de k=3. k=2 tem silhouette mais alto, mas só
separa dois grandes grupos, o que é pouco útil para recomendar por mood.
Silhouette próximo de 0.2 indica clusters sobrepostos, o que é normal em
audio features, que formam um contínuo.

## Estrutura

```
├── app.py                # interface Streamlit
├── src/
│   ├── model.py          # limpeza, clustering e exportação
│   └── evaluate_k.py     # silhouette e inércia para vários k
├── tests/                # testes com pytest (22)
└── data/                 # CSVs (ignorados pelo Git)
```

## Como usar

```bash
pip install -r requirements.txt

# 1) Coloque o CSV do Kaggle em data/dataset.csv e treine
python -m src.model --input-csv data/dataset.csv

# 2) Abra a app
python -m streamlit run app.py
```

Opcional:

```bash
python -m src.evaluate_k                 # avaliar outros valores de k
pip install -r requirements-dev.txt
python -m pytest -v                      # correr os testes
```

## Limitações

- O clustering usa só features acústicas: não captura idioma, época nem estilo
  (por isso o filtro por género melhora as recomendações).
- Clusters desequilibrados: um deles concentra cerca de um terço das faixas.
- Cada faixa mantém um único género (a primeira ocorrência no dataset).

## Ideias futuras

- Persistir `StandardScaler` e `KMeans` com `joblib`.
- Testar outros algoritmos (GMM, clustering hierárquico).
- Excluir versões alternativas da mesma música das recomendações.