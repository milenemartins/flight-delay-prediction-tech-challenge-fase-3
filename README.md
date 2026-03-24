# Tech Challenge Fase 3 — Machine Learning Engineering

> **POSTECH | Machine Learning Engineering**
> Análise preditiva de atrasos de voos nos EUA (dataset 2015 — 5,8 milhões de registros)

---

## Problema

O transporte aéreo é parte vital da infraestrutura global, mas atrasos impactam milhões de passageiros anualmente. Este projeto desenvolve um **pipeline completo de ciência de dados** — da exploração dos dados até modelos preditivos em produção — para analisar e prever atrasos de voos nos EUA.

---

## Estrutura do Projeto

```
tech_challenge_fase3/
├── data/
│   ├── raw/
│   │   ├── flights.csv          # Dataset principal (5,8M voos)
│   │   ├── airlines.csv         # 14 companhias aéreas
│   │   └── airports.csv         # 322 aeroportos (códigos IATA)
│   └── processed/
│       ├── flights_enriched.csv # EDA output — com nomes e coordenadas
│       ├── flights_processed.csv# Dados limpos + 18 features para ML
│       └── airport_clusters.csv # Resultado da clusterização K-Means
├── notebooks/
│   ├── 01_eda.ipynb                  # Análise Exploratória (EDA)
│   ├── 02_preprocessing.ipynb        # Pré-processamento + Feature Engineering
│   ├── 03_supervised_model.ipynb     # Classificação: vai atrasar?
│   ├── 04_unsupervised_model.ipynb   # Clusterização K-Means + PCA
│   ├── 05_regression_model.ipynb     # Regressão: quanto vai atrasar?
│   └── 06_two_stage_model.ipynb      # Pipeline dois estágios (diferencial)
├── models/
│   ├── random_forest_model.joblib        # RF Classifier
│   ├── xgboost_classifier.joblib         # XGBoost Classifier
│   ├── random_forest_regressor.joblib    # RF Regressor
│   ├── xgboost_regressor.joblib          # XGBoost Regressor
│   ├── pipeline_stage1_classifier.joblib # Two-Stage: Stage 1
│   ├── pipeline_stage2_regressor.joblib  # Two-Stage: Stage 2
│   └── scaler.joblib                     # StandardScaler
├── reports/
│   └── figures/                      # Gráficos e visualizações (PNG + HTML)
├── requirements.txt
└── README.md
```

---

## Requisitos do Projeto

### Obrigatório ✅
- [x] **EDA** — estatísticas descritivas, visualizações, tratamento de valores ausentes
- [x] **Modelagem Supervisionada** — ≥ 2 algoritmos com métricas adequadas
- [x] **Modelagem Não Supervisionada** — clusterização e/ou redução de dimensionalidade
- [x] **Apresentação crítica** — conclusões, limitações e próximos passos

### Diferenciais Implementados ⭐
- [x] Classificação **e** Regressão (ambas as abordagens supervisionadas)
- [x] Clusterização (K-Means) **e** PCA (ambas as abordagens não supervisionadas)
- [x] **Feature engineering avançado**: encoding cíclico (sin/cos), feriados americanos, histórico por rota, congestionamento do aeroporto
- [x] **Tratamento de desbalanceamento de classes** (`class_weight='balanced'` + `scale_pos_weight` + threshold ótimo)
- [x] **Cross-Validation** (5-Fold) em todos os modelos supervisionados
- [x] **XGBoost** como terceiro algoritmo em classificação e regressão
- [x] **Pipeline em dois estágios** — classifica → regride (notebook 06)
- [x] Mapas geográficos interativos (Plotly)
- [x] Curvas Precision-Recall e análise de threshold

---

## Notebooks — Visão Geral

### `01_eda.ipynb` — Análise Exploratória
- 5,8M voos analisados; 17,6% com atraso > 15 min
- 17 visualizações estáticas + 3 mapas interativos (HTML)
- Principais achados: horário de partida e companhia aérea são fatores críticos

### `02_preprocessing.ipynb` — Pré-processamento + Feature Engineering

**Limpeza:**
- Remoção de voos cancelados (89.884 registros)
- Tratamento de valores ausentes nas colunas de causa de atraso (fill = 0)

**18 features criadas:**

| Grupo | Features |
|-------|----------|
| Temporal cíclico | `MONTH_SIN/COS`, `DOW_SIN/COS`, `HOUR_SIN/COS` |
| Temporal ordinal | `DAY` |
| Operacional | `AIRLINE_ENCODED`, `ORIGIN_ENCODED`, `DEST_ENCODED` |
| Voo | `SCHEDULED_TIME`, `DISTANCE` |
| Derivadas | `PERIOD_ENCODED`, `SEASON_ENCODED`, `IS_WEEKEND` |
| Avançadas | `IS_HOLIDAY`, `ROUTE_DELAY_MEAN`, `ORIGIN_DAILY_FLIGHTS` |

### `03_supervised_model.ipynb` — Classificação

**Pergunta: "O voo vai atrasar (> 15 min)?"**

| Modelo | Accuracy | F1-Score | ROC-AUC |
|--------|----------|----------|---------|
| Logistic Regression | ~74% | >0 | ~0.58 |
| Random Forest | ~74% | >0 | ~0.68 |
| **XGBoost** | **melhor** | **melhor** | **~0.70+** |
| RF + threshold ótimo | varia | melhor F1 | ~0.68 |

> Valores exatos gerados na execução. Desbalanceamento tratado com `class_weight='balanced'` e `scale_pos_weight`.

**Adicionalmente:**
- Curvas ROC e Precision-Recall para os 3 modelos
- Ajuste de threshold para maximizar F1-Score
- Cross-Validation 5-Fold (amostra 200k)

### `04_unsupervised_model.ipynb` — Clusterização + PCA

**K-Means (K=4) em 297 aeroportos:**

| Cluster | Label | Qtd | Atraso Médio | Taxa Atraso |
|---------|-------|-----|--------------|-------------|
| 0 | Aeroportos Eficientes | ~91 | -1,0 min | 11,2% |
| 1 | Regionais Médios | ~108 | 3,2 min | 15,5% |
| 2 | Regionais Problemáticos | ~67 | 7,2 min | 18,7% |
| 3 | Grandes Hubs | ~32 | 7,1 min | 21,3% |

**PCA:** 2 componentes explicam 87% da variância.

> 73 aeroportos usam códigos BTS numéricos (sem correspondência IATA) — participam do clustering mas não aparecem no mapa geográfico.

### `05_regression_model.ipynb` — Regressão

**Pergunta: "Quantos minutos de atraso?"**

| Modelo | MAE | RMSE | R² |
|--------|-----|------|----|
| Linear Regression | 28,0 min | 41,7 min | 0,011 |
| Random Forest | 26,2 min | 39,7 min | 0,107 |
| **XGBoost** | **menor** | **menor** | **maior** |

> Valores exatos do XGBoost gerados na execução. Cross-Validation 5-Fold confirma estabilidade.

### `06_two_stage_model.ipynb` — Pipeline em Dois Estágios ⭐

**Combina classificação e regressão em um único pipeline realista:**

```
Voo → [Stage 1: XGBoost Classifier] → Vai atrasar?
              ├── NÃO → Previsão = 0 min
              └── SIM → [Stage 2: XGBoost Regressor] → Previsão = X min
```

**Vantagens sobre regressão direta:**
- Gera previsão de minutos para **qualquer voo** (não só os já atrasados)
- Separação de responsabilidades entre os dois subproblemas
- Redução de ruído no Stage 2 (treinado apenas em voos com atraso real)

---

## Como Executar

### 1. Ambiente

```bash
# Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Instalar dependências
pip install -r requirements.txt

# macOS: instalar dependência do XGBoost
brew install libomp
```

### 2. Dataset

Baixar o dataset do Kaggle e colocar os arquivos em `data/raw/`:
- `flights.csv`
- `airlines.csv`
- `airports.csv`

### 3. Executar notebooks na ordem

```
01_eda.ipynb              → gera flights_enriched.csv
02_preprocessing.ipynb    → gera flights_processed.csv (com 18 features)
03_supervised_model.ipynb → classificação + XGBoost
04_unsupervised_model.ipynb → clustering + PCA
05_regression_model.ipynb → regressão + XGBoost
06_two_stage_model.ipynb  → pipeline dois estágios (diferencial)
```

---

## Dependências

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
xgboost>=2.0.0
holidays>=0.8
jupyter>=1.0.0
ipykernel>=6.25.0
tqdm>=4.65.0
joblib>=1.3.0
```

---

## Dataset

| Arquivo | Tamanho | Descrição |
|---------|---------|-----------|
| `flights.csv` | 565 MB | 5.819.079 voos nos EUA (2015) |
| `airlines.csv` | <1 KB | 14 companhias aéreas (códigos IATA) |
| `airports.csv` | 23 KB | 322 aeroportos com coordenadas |

**Fonte**: [Flight Delays and Cancellations — Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)

---

## Entregáveis

- [x] Repositório GitHub com código completo
- [x] Vídeo de apresentação (5–10 minutos)
