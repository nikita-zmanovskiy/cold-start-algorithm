# Quick Start: Finalization Steps

## Критические шаги для финализации paper

### Шаг 1: Исправить GT (если есть missing items)

**Вариант 1: Автоматическая замена (рекомендуется)**
```bash
python -m tools.fix_gt_catalog --replace
```
Это автоматически:
- Создаст backup оригинального `ground_truth.json`
- Заменит его на исправленную версию

**Вариант 2: Ручная замена**
```bash
# 1. Сначала исправить GT
python -m tools.fix_gt_catalog

# 2. Проверить отчет
cat experiments/gt_fix_report.json

# 3. Вручную заменить (Windows PowerShell):
Copy-Item experiments/ground_truth.json experiments/ground_truth_original_backup.json
Copy-Item experiments/ground_truth_fixed.json experiments/ground_truth.json

# Или в Linux/Mac:
cp experiments/ground_truth.json experiments/ground_truth_original_backup.json
cp experiments/ground_truth_fixed.json experiments/ground_truth.json
```

**Что происходит:**
- Скрипт находит GT items, которых нет в каталоге
- Удаляет пользователей, у которых ВСЕ GT items отсутствуют в каталоге
- Для остальных пользователей удаляет только missing items, оставляя валидные
- Создает `ground_truth_fixed.json` с исправленными данными

### Шаг 2: Пересобрать эксперименты (с исправленным GT и сохранением scores)

```bash
python -m src.run_all_experiments --n-users 500 --seeds 42 7 123
```

### Шаг 3: Запустить ablation по pool sizes

```bash
python -m src.run_ablation_pool_sizes --pool-sizes 200 500 1000 --seeds 42 7 123
```

### Шаг 4: Собрать все данные

```bash
python -m tools.build_master_results
python -m tools.aggregate_runs
```

### Шаг 5: Провести все анализы

```bash
python -m tools.hypothesis_analysis
python -m tools.analyze_scores
python -m tools.error_analysis
python -m tools.enhanced_stat_tests
```

### Шаг 6: Сгенерировать визуализации

```bash
python -m tools.plotting
python -m tools.advanced_plotting
```

### Шаг 7: Создать таблицы для статьи

```bash
python -m tools.generate_paper_tables
```

## Результаты

После выполнения всех шагов у вас будет:

### Файлы данных:
- `experiments/master_results.json` - полные per-user данные
- `experiments/aggregated_results.csv/json` - агрегированные метрики
- `experiments/stat_tests/comprehensive_stat_test_results.json` - статистические тесты

### Визуализации:
- `experiments/plots/*.png` - основные графики
- `experiments/hypothesis_analysis/*.png` - анализ гипотез
- `experiments/score_analysis/*.png` - анализ scores

### Таблицы:
- `experiments/tables/main_results.tex` - LaTeX таблица
- `experiments/tables/main_results.md` - Markdown таблица
- `experiments/tables/diagnostics.md` - диагностика

### Error Analysis:
- `experiments/error_analysis/error_analysis_examples.json` - примеры ошибок

## Что включить в статью

1. **Results Section:**
   - Таблица с HR@10, nDCG@10 (mean ± std ± CI)
   - Таблица статистических тестов
   - Таблица ablation (pool sizes)

2. **Figures:**
   - Recall@K curves
   - Boxplots per-user metrics
   - Top-1 exposure plots
   - Score distributions
   - GT positions histogram

3. **Diagnostics:**
   - Recall@K values
   - Unique top-1, Gini, entropy
   - Exposure metrics

4. **Error Analysis:**
   - 5-10 примеров с объяснениями

5. **Limitations:**
   - Обсуждение проблем и причин

6. **Appendix:**
   - Master results JSON
   - Per-user CSVs
   - Reproduction instructions
