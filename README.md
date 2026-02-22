# Cold-Start Recommendation Algorithm

Research-grade recommendation system for cold-start scenarios.

## ✅ Current Status

- ✔ **Real ground-truth data** from Serendipity-2018 dataset
- ✔ **Correct evaluation pipeline** (HR@10, NDCG@10)
- ✔ **Baseline methods** (Random, Popularity, Embedding cosine)
- ✔ **Unified experiment logging** (runs.jsonl)
- ✔ **Ablation study support**
- ✔ **Canonical evaluation protocol** — 500 test users, 5 seeds; see [experiments/EVALUATION_PROTOCOL.md](experiments/EVALUATION_PROTOCOL.md) for the paper

## 🚀 Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Preprocess data (if not done)
python -m src.preprocess

# Create ground truth (if not exists)
python -m src.create_ground_truth --csv data/processed/interactions.csv --out experiments/ground_truth.json
```

### 3. Run Experiments

**Run all experiments (baselines + ablation)** — uses 500 test users and 5 seeds by default (see `src/evaluation_config.py`):
```bash
python -m src.run_all_experiments
```

**Run only baselines:**
```bash
python -m src.run_all_experiments --baselines-only
```

**Run only ablation study:**
```bash
python -m src.run_all_experiments --ablation-only
```

**Run pool size ablation:**
```bash
python -m src.run_ablation_pool_sizes --pool-sizes 200 500 1000
```

To override: `--n-users N --seeds 42 7 123 2024 2025`

### 4. Full pipeline (one command)

Запустить весь исследовательский пайплайн (эксперименты, агрегация, гипотезы, визуализации, статтесты) одной командой:

```bash
python -m tools.full_pipeline
```

В конце скрипт напечатает, какие артефакты были созданы и в каких папках они лежат (master_results, aggregated_results, plots, hypothesis_analysis, score_analysis, stat_tests и т.д.).

### 5. View Results

All experiments are logged to `experiments/runs.jsonl` (JSON Lines format).

**Aggregate results:**
```bash
python -m tools.aggregate_runs
```

**Build master results (per-user data + CI):**
```bash
python -m tools.build_master_results
```

**Generate visualizations:**
```bash
python -m tools.plotting
python -m tools.advanced_plotting
```

**Statistical tests:**
```bash
python -m tools.stat_tests
python -m tools.enhanced_stat_tests
```

**Generate paper tables:**
```bash
python -m tools.generate_paper_tables
```

See `ANALYSIS_TOOLS.md` for detailed documentation.

## 📊 Experiment Configurations

### Baselines

1. **Random**: Random item selection
2. **Popularity**: Most popular items
3. **Embedding cosine**: Cosine similarity between user profile and items (no reranker)

### Ablation Study

1. **candidates_only**: FAISS retrieval only (no reranker)
2. **with_reranker**: FAISS + Cross-encoder reranker

## 📁 Project Structure

```
src/
├── baselines.py          # Baseline methods (Random, Popularity, Embedding)
├── run_experiment.py     # Main experiment runner
├── run_all_experiments.py # Run all experiments script
├── run_logger.py         # Unified experiment logging (runs.jsonl)
├── evaluate_results.py   # Evaluation metrics (HR@10, NDCG@10)
├── candidate_retrieval.py # FAISS-based candidate retrieval
├── rerank_llm.py         # Cross-encoder reranker
└── ...

experiments/
├── runs.jsonl           # Unified experiment log (JSON Lines)
├── ground_truth.json     # Real GT data
└── eval_*.csv           # Per-user evaluation results

results/
└── *.json               # Raw experiment results
```

## 🔬 Research Workflow

1. **Run experiments** → logged to `runs.jsonl`
2. **Analyze results** → load from `runs.jsonl`
3. **Generate tables/plots** → from unified log
4. **Answer reviewers** → all runs are logged automatically

## 📝 Notes

- **No simulation**: Uses real ground-truth data only
- **Cold-start**: User profiles are minimal (no history)
- **Reproducible**: All runs logged with seed and config
- **Scalable**: Supports 500+ users

## 🧹 Cleanup

Remove demo/test files:
```bash
python tools/cleanup_demo_files.py
```

Dry-run first:
```bash
python tools/cleanup_demo_files.py --dry-run
```



Следующие шаги
Пересобрать эксперименты:
   python -m src.run_all_experiments --n-users 500 --seeds 42 7 123
Агрегировать результаты:
   python -m tools.aggregate_runs
Сгенерировать визуализации:
   python -m tools.plotting
Создать таблицы для статьи:
   python -m tools.generate_paper_tables
Запустить статистические тесты:
   python -m tools.stat_tests



# 1. Соберите master results (per-user arrays + CI)
python -m tools.build_master_results

# 2. Сгенерируйте расширенные визуализации
python -m tools.advanced_plotting

# 3. Запустите расширенные статистические тесты
python -m tools.enhanced_stat_tests

# 4. Проверьте гипотезы о причинах проблем
python -m tools.hypothesis_analysis

# 5. Анализ reranker scores
python -m tools.analyze_scores

# 6. Error analysis
python -m tools.error_analysis

# 7. Исправить GT (если есть missing items)
# Автоматически создаст backup и заменит файл:
python -m tools.fix_gt_catalog --replace

# Или только исправить без замены (потом заменить вручную):
python -m tools.fix_gt_catalog







# 1. Исправить GT
python -m tools.fix_gt_catalog
# Затем заменить ground_truth.json на ground_truth_fixed.json

# 2. Пересобрать эксперименты
python -m src.run_all_experiments --n-users 500 --seeds 42 7 123

# 3. Ablation по pool sizes
python -m src.run_ablation_pool_sizes --pool-sizes 200 500 1000

# 4. Собрать все данные
python -m tools.build_master_results
python -m tools.aggregate_runs

# 5. Все анализы
python -m tools.hypothesis_analysis
python -m tools.analyze_scores
python -m tools.error_analysis
python -m tools.enhanced_stat_tests

# 6. Визуализации
python -m tools.plotting
python -m tools.advanced_plotting

# 7. Таблицы
python -m tools.generate_paper_tables





python -m tools.full_pipeline --clean --fast --rebuild-gt

python -m tools.full_pipeline --clean --fast --rebuild-gt