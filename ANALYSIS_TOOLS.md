# Advanced Analysis Tools

## Новые инструменты для детального анализа результатов

### 1. Master Results Builder

Собирает все per-user данные в единый master JSON файл с:
- Per-user arrays для HR@10 и nDCG@10
- Bootstrap confidence intervals (95%)
- Per-item exposure statistics
- Полная диагностика на уровне пользователя

```bash
python -m tools.build_master_results
```

Создает: `experiments/master_results.json`

### 2. Advanced Plotting

Генерирует расширенные визуализации:
- Детальные recall curves (K до 1000)
- Boxplots per-user HR@10 и nDCG@10
- Top-1 concentration plots
- Item exposure distribution

```bash
python -m tools.advanced_plotting
```

Требует: `experiments/master_results.json` (запустите сначала build_master_results)

Создает в `experiments/plots/`:
- `recall_curves_detailed.png`
- `boxplots_per_user.png`
- `top1_concentration.png`
- `item_exposure_distribution.png`

### 3. Enhanced Statistical Tests

Расширенные статистические тесты с:
- Bootstrap confidence intervals для mean difference
- Effect size interpretation (small/medium/large)
- Полные статистики (t-test, Wilcoxon, Cohen's d)

```bash
python -m tools.enhanced_stat_tests
```

Требует: `experiments/master_results.json`

Создает: `experiments/stat_tests/comprehensive_stat_test_results.json`

## Порядок использования

1. **Соберите master results:**
   ```bash
   python -m tools.build_master_results
   ```

2. **Сгенерируйте визуализации:**
   ```bash
   python -m tools.advanced_plotting
   ```

3. **Запустите расширенные статистические тесты:**
   ```bash
   python -m tools.enhanced_stat_tests
   ```

## Структура master_results.json

```json
{
  "meta": {
    "project": "cold-start-algorithm",
    "dataset": "Serendipity-2018",
    "n_runs": 16
  },
  "runs": [
    {
      "run_id": "...",
      "config": {...},
      "metrics": {
        "hr@10": {
          "mean": 0.074,
          "std": 0.262,
          "ci_95_lower": 0.045,
          "ci_95_upper": 0.103,
          "per_user": [0.0, 1.0, 0.0, ...],
          "n_users": 500
        },
        ...
      },
      "per_user_detail": {
        "user_id": {
          "hr@10": 0.0,
          "ndcg@10": 0.0,
          "rec_ids": "...",
          "hits": "...",
          "hit_bool": 0
        }
      }
    }
  ],
  "per_item_exposure": {
    "item_id": {
      "exposure_count": 150,
      "unique_users": 120
    }
  }
}
```

### 4. Hypothesis Analysis

Проверяет ключевые гипотезы о причинах низкой производительности:
- Coverage/Retrieval: recall@K curves, распределение позиций GT
- Bias/Exposure: Gini, entropy, top-20 exposure plots
- Score distributions: (требует сохранения scores в run_experiment)
- Recall vs HR correlation: scatter plots
- GT/Catalog quality: проверка missing items

```bash
python -m tools.hypothesis_analysis
```

Требует: `experiments/master_results.json`

Создает в `experiments/hypothesis_analysis/`:
- `coverage_recall_curves.png`
- `gt_positions_histogram.png`
- `top1_exposure_*.png`
- `recall_vs_hr_*.png`

## Что дальше

После сбора master results вы можете:
- Анализировать per-user patterns
- Вычислять корреляции (recall vs HR@10)
- Проводить error analysis на конкретных пользователях
- Изучать item exposure и long-tail distribution
- Проверять гипотезы о причинах проблем
