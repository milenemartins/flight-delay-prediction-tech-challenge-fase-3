# Tech Challenge Fase 3 - Machine Learning Engineering

## Objetivo
Análise de atrasos de voos nos EUA utilizando técnicas de Machine Learning supervisionado e não supervisionado.

## Estrutura do Projeto

```
tech_challenge_fase3/
├── data/
│   ├── raw/                 # Dados originais (flights.csv)
│   └── processed/           # Dados processados
├── notebooks/
│   ├── 01_eda.ipynb                    # Análise Exploratória
│   ├── 02_preprocessing.ipynb          # Pré-processamento
│   ├── 03_supervised_model.ipynb       # Classificação (prever SE vai atrasar)
│   ├── 04_unsupervised_model.ipynb     # Clusterização + PCA
│   └── 05_regression_model.ipynb       # Regressão (prever QUANTO vai atrasar)
├── src/
│   └── __init__.py
├── models/                  # Modelos treinados salvos
├── reports/
│   └── figures/             # Gráficos e visualizações
├── requirements.txt
└── README.md
```

## Requisitos do Projeto

### Obrigatório
- [x] EDA (Análise Exploratória de Dados)
- [x] Modelagem Supervisionada (mínimo 2 algoritmos)
- [x] Modelagem Não Supervisionada (clusterização ou PCA)
- [x] Apresentação crítica dos resultados

### Diferenciais Implementados
- [x] Classificação E Regressão (ambas abordagens supervisionadas)
- [x] Clusterização E PCA (ambas abordagens não supervisionadas)
- [x] Variáveis derivadas (período do dia, fim de semana, estações)

## Modelos Implementados

### Supervisionado - Classificação (notebook 03)
Prever **SE** um voo vai atrasar (atraso > 15 min)
- Logistic Regression
- Random Forest Classifier

### Supervisionado - Regressão (notebook 05)
Prever **QUANTO TEMPO** o atraso vai durar (em minutos)
- Linear Regression
- Random Forest Regressor

### Não Supervisionado (notebook 04)
- K-Means (clusterização de aeroportos)
- PCA (redução de dimensionalidade)

## Como Executar

1. Instalar dependências:
```bash
pip install -r requirements.txt
```

2. Baixar o dataset e colocar em `data/raw/flights.csv`

3. Executar os notebooks na ordem numérica:
   - 01 → 02 → 03 → 04 → 05

## Dataset
- **Fonte**: Flight Delays and Cancellations (Kaggle)
- **Colunas principais**: DEPARTURE_DELAY, ARRIVAL_DELAY, AIRLINE, ORIGIN_AIRPORT, etc.

## Entregáveis
- Repositório GitHub com código completo
- Vídeo de apresentação (5-10 minutos)
