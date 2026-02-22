# Исправления экспериментов для статьи

## ✅ Что было исправлено

### 1. Методологическая ошибка: use_reranker у baseline'ов

**Проблема:** Baseline методы (random, popularity, embedding_cosine) использовали `use_reranker=true`, что нечестно для сравнения.

**Исправление:**
- В `src/run_all_experiments.py` все baseline методы теперь используют `use_reranker=False`
- Правило: **Baseline = без reranker ВСЕГДА**

### 2. Отсутствие recall@K в диагностике

**Проблема:** `gt_recall@1000` был `null` во всех запусках, что критично для статьи.

**Исправление:**
- В `src/run_logger.py` добавлено вычисление `recall@50`, `recall@200`, `recall@1000`
- В `src/run_experiment.py` добавлено отслеживание candidate pools для каждого пользователя
- Диагностика теперь включает:
  - `recall@50`, `recall@200`, `recall@1000`
  - `unique_top1`, `top1_counts` (топ-10)
  - `topk_length_distribution`
  - `rerank_time_mean/std`

### 3. Агрегация по seed'ам

**Добавлено:**
- `tools/aggregate_runs.py` - агрегирует результаты по конфигурациям с mean ± std
- Генерирует `experiments/aggregated_results.csv` и `experiments/aggregated_results.json`

### 4. Инструменты для статьи

**Созданы:**
- `tools/plotting.py` - визуализации:
  - Bar chart HR@10 ± std
  - Bar chart nDCG@10 ± std
  - Recall@K curves
  - Histogram unique_top1 (bias analysis)
- `tools/generate_paper_tables.py` - таблицы для статьи:
  - LaTeX таблица
  - Markdown таблицы (основные метрики + диагностика)
- `tools/stat_tests.py` - статистические тесты:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Cohen's d (effect size)

## 📋 Как использовать

### Шаг 1: Пересобрать baseline эксперименты

```bash
# Удалить старые результаты (опционально)
# rm experiments/runs.jsonl

# Запустить все эксперименты заново (baselines БЕЗ reranker)
python -m src.run_all_experiments --n-users 500 --seeds 42 7 123
```

Это создаст новые записи в `runs.jsonl` с правильными конфигурациями:
- `baseline_random` → `use_reranker: false`
- `baseline_popularity` → `use_reranker: false`
- `baseline_embedding_cosine` → `use_reranker: false`
- `ablation_candidates_only` → `use_reranker: false`
- `ablation_with_reranker` → `use_reranker: true` ✅

### Шаг 2: Агрегировать результаты

```bash
python -m tools.aggregate_runs
```

Создаст:
- `experiments/aggregated_results.csv`
- `experiments/aggregated_results.json`

### Шаг 3: Сгенерировать визуализации

```bash
python -m tools.plotting
```

Создаст в `experiments/plots/`:
- `hr10_bar_chart.png`
- `ndcg10_bar_chart.png`
- `recall_curves.png`
- `unique_top1_histogram.png`

### Шаг 4: Сгенерировать таблицы для статьи

```bash
python -m tools.generate_paper_tables
```

Создаст в `experiments/tables/`:
- `main_results.tex` (LaTeX)
- `main_results.md` (Markdown)
- `diagnostics.md` (Markdown)

### Шаг 5: Статистические тесты

```bash
python -m tools.stat_tests
```

Создаст `experiments/stat_tests/stat_test_results.json` с p-values и effect sizes.

## 📊 Структура runs.jsonl

Каждая строка теперь содержит:

```json
{
  "run_id": "baseline_random_seed42_n500",
  "timestamp": "2026-02-06T...",
  "config": {
    "seed": 42,
    "n_users": 500,
    "topk": 10,
    "baseline": "random",
    "use_reranker": false,  // ✅ Исправлено
    "candidate_pool_size": 1000
  },
  "metrics": {
    "hr@10": {"mean": 0.0, "std": 0.0},
    "ndcg@10": {"mean": 0.0, "std": 0.0}
  },
  "diagnostics": {
    "recall@50": 0.20,      // ✅ Добавлено
    "recall@200": 0.70,     // ✅ Добавлено
    "recall@1000": 0.95,    // ✅ Добавлено
    "unique_top1": 1,
    "top1_counts": [["164999", 500]],
    "topk_length_distribution": {"10": 500},
    "rerank_time_mean": null,
    "rerank_time_std": null
  },
  "files": {
    "raw_results": "results/baseline_random_seed42_n500.json"
  }
}
```

## 🔍 Проверка результатов

После пересборки проверьте:

1. **Baseline без reranker:**
   ```bash
   grep '"baseline": "random"' experiments/runs.jsonl | grep '"use_reranker": false'
   ```

2. **Recall@K заполнен:**
   ```bash
   grep '"recall@1000"' experiments/runs.jsonl | head -1
   ```
   Должно быть число (например, `0.95`), а не `null`.

3. **Агрегированные результаты:**
   ```bash
   cat experiments/aggregated_results.csv
   ```

## ⚠️ Важные замечания

1. **Старые результаты в runs.jsonl** - они содержат ошибки. Либо удалите файл и пересоберите, либо отфильтруйте старые записи.

2. **Baseline методы** - для них candidate pool = все items (для recall@K вычисления). Это нормально, т.к. они не используют retrieval.

3. **Reranker bias** - если `unique_top1 = 1`, это нормально, но должно быть явно описано в статье как диагностика.

4. **Статистические тесты** - требуют per-user CSV файлы. Убедитесь, что они создаются при запуске экспериментов.

## 📝 Для статьи

Используйте:
- `experiments/tables/main_results.tex` - основная таблица
- `experiments/tables/diagnostics.md` - диагностика (recall@K, bias)
- `experiments/plots/*.png` - фигуры
- `experiments/stat_tests/stat_test_results.json` - статистические тесты
