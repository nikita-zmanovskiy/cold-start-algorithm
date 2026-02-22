# Hypothesis Testing Framework

## Инструмент для проверки гипотез

`tools/hypothesis_analysis.py` автоматически проверяет ключевые гипотезы о причинах проблем с производительностью.

## Запуск

```bash
python -m tools.hypothesis_analysis
```

Требует:
- `experiments/master_results.json` (запустите `build_master_results` сначала)
- `results/*.json` файлы с candidate pools

## Проверяемые гипотезы

### 1. Coverage / Retrieval (Недостаточное покрытие)

**Гипотеза:** Релевантные GT элементы не попадают в пул кандидатов.

**Измерения:**
- Recall@K для K = 50, 200, 500, 1000 (per-user и среднее)
- Распределение позиций GT в candidate pool (median, Q1, Q3)
- Recall curves для всех моделей

**Выход:**
- `coverage_recall_curves.png` - кривые recall@K
- `gt_positions_histogram.png` - распределение позиций GT

**Интерпретация:**
- Если recall@200 < 0.5 → проблема в retrieval
- Если median position > 200 → GT слишком глубоко в pool

### 2. Bias Reranker / Exposure (Концентрация на популярных)

**Гипотеза:** Reranker концентрирует выбор на небольшом наборе items.

**Измерения:**
- Unique top-1 items
- Gini coefficient (концентрация)
- Entropy (разнообразие)
- Top-20 most frequent top-1 items

**Выход:**
- `top1_exposure_*.png` - барплоты топ-20 для каждой модели

**Интерпретация:**
- Gini > 0.7 → сильная концентрация
- Unique top-1 < 10 → reranker biased
- Если top-1 совпадает с популярными → reranker усиливает popularity

### 3. Score Distributions / Calibration

**Гипотеза:** Проблемы с масштабом/калибровкой scores.

**Статус:** Требует сохранения scores в run_experiment.py

**Планируемые измерения:**
- Распределения scores для релевантных vs нерелевантных
- Корреляция score → relevance (Spearman)

### 4. Recall vs HR Correlation

**Гипотеза:** Корреляция между recall в candidates и HR@10.

**Измерения:**
- Per-user recall@200 в candidates (binary)
- Per-user HR@10
- Корреляция Пирсона

**Выход:**
- `recall_vs_hr_*.png` - scatter plots

**Интерпретация:**
- Высокая корреляция (>0.5) → recall критичен для HR
- Низкая корреляция → проблема в reranker

### 5. GT / Catalog Quality

**Гипотеза:** Отсутствие GT items в каталоге искажает метрики.

**Измерения:**
- Список missing GT items
- Количество пользователей с missing items
- Процент missing items

**Интерпретация:**
- Если >5% missing → нужно исправить GT/catalog
- Если много пользователей затронуто → пересчитать метрики

## Пример вывода

```
============================================================
HYPOTHESIS 1: Coverage / Retrieval Analysis
============================================================

Recall@K by Model (Mean):

  candidates_only:
    Recall@50: 0.234 (n=500)
    Recall@200: 0.567 (n=500)
    Recall@500: 0.789 (n=500)
    Recall@1000: 0.923 (n=500)

GT Position in Candidates:
  Median: 145.0
  Q1: 67.0
  Q3: 312.0
  Mean: 198.3

============================================================
HYPOTHESIS 2: Bias Reranker / Exposure Analysis
============================================================

ours_with_reranker:
  Unique top-1 items: 3
  Gini coefficient: 0.847
  Entropy: 1.234
  Top-5 most frequent top-1:
    175353: 245 (0.490)
    702: 129 (0.258)
    1035: 126 (0.252)
```

## Рекомендации по результатам

### Если Hypothesis 1 подтверждается (низкий recall):
1. Увеличить retrieval_top_k
2. Улучшить embedding model
3. Добавить hybrid retrieval (embedding + popularity)
4. Использовать graph expansion

### Если Hypothesis 2 подтверждается (высокий bias):
1. Rerank только top-200 вместо top-1000
2. Добавить diversity penalty (MMR)
3. Калибровать scores
4. Fine-tune reranker

### Если Hypothesis 4 показывает низкую корреляцию:
- Проблема не в retrieval, а в reranker
- Нужно улучшить reranker или отключить его

### Если Hypothesis 5 показывает missing items:
- Исправить GT/catalog перед дальнейшими экспериментами
- Пересчитать все метрики
