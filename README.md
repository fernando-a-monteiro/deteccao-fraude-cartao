# Detecção de Fraude em Cartão de Crédito

Modelos de classificação para detectar fraude em transações reais de cartão, num cenário em que **apenas 0,17% das transações são fraude**. Projeto do desafio DIO.

📓 Notebook: [`deteccao_fraude_cartao.ipynb`](deteccao_fraude_cartao.ipynb)

## 1. O problema e por que o desbalanceamento muda a avaliação

A base tem 284.807 transações de cartão, com as colunas `Time`, `Amount`, `Class` e `V1` a `V28` (componentes de PCA, anonimizadas). Só 492 são fraude.

Um "modelo" que responde **sempre "não é fraude"** tem **99,83% de acurácia** e não detecta nenhuma fraude. Por isso a avaliação usa as métricas da **classe fraude**:

- **Recall (métrica principal):** das fraudes reais, quantas foram pegas.
- **Precisão:** dos alertas, quantos eram fraude de verdade.
- **F1:** equilíbrio entre os dois.
- **PR-AUC:** qualidade do ranking de probabilidades, sem depender de limiar. Neste problema, a ROC-AUC fez a Regressão Logística (0,961) parecer melhor que o Random Forest (0,938), mas a PR-AUC mostrou o contrário (0,680 contra 0,807).

## 2. Preparação dos dados

- Remoção de **1.081 linhas duplicadas**, para que a mesma transação não caia no treino e no teste.
- Novas variáveis: `Amount_log = log1p(Amount)` e `Hour = (Time // 3600) % 24`.
- Divisão **treino / validação / teste (60/20/20) com `stratify`**. O teste ficou com 95 fraudes.
- `StandardScaler` dentro de um `Pipeline` na Regressão Logística, para que a padronização aprenda só com o treino.
- Desbalanceamento tratado com **peso das classes** (`class_weight` / `scale_pos_weight`).

## 3. Comparação dos modelos (classe fraude, conjunto de teste)

| Modelo | Limiar | Recall | Precisão | F1 | PR-AUC |
|---|---|---|---|---|---|
| Dummy (sempre normal) | 0,5 | 0,000 | 0,000 | 0,000 | 0,002 |
| Regressão Logística | 0,9998 | 0,768 | 0,777 | 0,773 | 0,680 |
| Random Forest | 0,220 | 0,779 | 0,914 | 0,841 | 0,807 |
| XGBoost | 0,931 | 0,737 | 0,986 | 0,843 | 0,828 |
| **XGBoost otimizado (final)** | **0,673** | **0,758** | **0,973** | **0,852** | **[PR-AUC]** |

**Undersampling × SMOTE** (na Regressão Logística): o undersampling foi o pior (PR-AUC de 0,323, 57 fraudes pegas), porque descarta 99,8% das normais. O SMOTE empatou com o `class_weight` (73 fraudes pegas nos dois), então o `class_weight` foi mantido por ser mais simples.

**GridSearchCV** (8 combinações, 3 folds, otimizando PR-AUC): melhor combinação `learning_rate=0.1`, `max_depth=5`, `n_estimators=400`.

## 4. Limiar de decisão e o que o SHAP mostrou

**Limiar:** escolhido **no conjunto de validação**, maximizando o **F2** (que dá peso dobrado ao recall). O teste foi usado só no final.

- Com peso de classes, as probabilidades do modelo ficam infladas, e o 0,5 perde o sentido. Os limiares ficaram em 0,9998 (Logística), 0,22 (Random Forest) e 0,93 (XGBoost).
- O ajuste **melhorou o Random Forest** (67 → 74 fraudes pegas) e **tornou a Logística utilizável** (alertas falsos de 1.591 para 21).
- No **XGBoost, o ajuste não generalizou**: o recall no teste caiu de 0,79 para 0,74. A validação tem só 94 fraudes, o que deixa a escolha do limiar instável.

**Modelo final:** XGBoost otimizado, limiar 0,673. Detectou **72 de 95 fraudes** com **2 alertas falsos**. A escolha foi um empate técnico com o Random Forest (F2 na validação de 0,8766 contra 0,8761).

**SHAP:**
- **V14 baixo** é o sinal mais forte de fraude, seguido de **V4 alto** e de **V12, V3 e V10 baixos**.
- `Amount_log` aparece em 12º: valores maiores empurram um pouco para fraude.
- As fraudes que escapam são as que têm **V14 com valor normal**. No exemplo analisado, V4 apontou fraude, mas V14 = −0,33 puxou a decisão de volta para normal.

## 5. O que mudei em relação à Expert

1. **Três conjuntos (treino/validação/teste):** limiar e modelo final escolhidos na validação, para que a nota do teste não seja otimista.
2. **Critério explícito para o limiar (F2)**, coerente com o recall como métrica principal.
3. **Remoção de duplicadas** antes da divisão.
4. **Variável `Hour`** derivada de `Time`.
5. **GridSearchCV otimizando PR-AUC**, e não acurácia.
6. **SHAP aplicado a uma fraude que escapou**, além de uma detectada.
7. **Registro dos resultados negativos:** o ajuste de limiar que não generalizou no XGBoost e o empate técnico na escolha do modelo final.

## Limitações e próximos passos

- Escolher o limiar com validação cruzada (`cross_val_predict`) em treino + validação, para usar ~378 fraudes em vez de 94.
- Ampliar a grade do GridSearch, já que os melhores valores ficaram nas bordas.
- A base não tem identificador de cartão, então não é possível criar variáveis de comportamento por cartão.

## Como executar

```bash
pip install -r requirements.txt
```

Abra `deteccao_fraude_cartao.ipynb` e rode todas as células. O dataset é baixado pelo link definido em `DATA_URL` e **não fica no repositório**.

## Tecnologias

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · SHAP · Matplotlib