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

## Comparação com GMM

Testei `GaussianMixture` (covariâncias `diag` e `full`) nas mesmas features
padronizadas. Em k=5, o silhouette foi 0.042 (`diag`) e -0.004 (`full`), contra
0.222 do K-Means, e o ARI entre os dois métodos foi 0.337 e 0.200. O BIC desce
até k=10 sem cotovelo, por isso não indica um k, e as probabilidades são
demasiado extremas (menos de 3 % de faixas ambíguas) para servirem como
medida de confiança. Mantive o K-Means com k=5.

## Transformação das features

Testei binarizar `instrumentalness` e aplicar Yeo-Johnson antes do K-Means (k=5).
A binarização deu um silhouette de 0.225 (baseline 0.218) e ARI 0.950 face ao
baseline, ou seja, quase os mesmos clusters. O Yeo-Johnson alterou mais os
grupos (ARI 0.704) e equilibrou um pouco os tamanhos (8 663 a 25 979 faixas, em
vez de 6 739 a 29 998), mas com silhouette 0.207. Como o silhouette não é
comparável entre espaços transformados e os ganhos são pequenos, mantive o
`StandardScaler` simples, que também simplifica a persistência do modelo.

## Avaliação das recomendações

Em 200 faixas de partida (filtro de género desligado), medi a percentagem de
recomendações com o mesmo género da faixa escolhida. A ordenação por distância
ficou claramente acima do baseline aleatório (aprox. 9 % contra 1.6 % no cluster
e 0.8 % no dataset). Dar mais peso a features individuais não alterou de forma
mensurável esta métrica. A opção "preferir populares" quase duplicou a
popularidade média das sugestões (33 para 61), com uma descida não conclusiva
da consistência de género. O género é um proxy ruidoso (114 géneros, um por
faixa), por isso os valores absolutos são baixos.

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

**App online:** https://persona-vibe-engine-wrs26wkrbfjbtzbqzr48s8.streamlit.app

> A app publicada usa uma amostra de 30 000 faixas (`data/dataset.csv`).
> As métricas deste README foram calculadas sobre o dataset completo.

## Limitações

- O clustering usa só features acústicas: não captura idioma, época nem estilo
  (por isso o filtro por género melhora as recomendações).
- Clusters desequilibrados: um deles concentra cerca de um terço das faixas.
- Cada faixa mantém um único género (a primeira ocorrência no dataset).

## Ideias futuras

- Persistir `StandardScaler` e `KMeans` com `joblib`.
- Testar outros algoritmos (GMM, clustering hierárquico).
- Excluir versões alternativas da mesma música das recomendações.
