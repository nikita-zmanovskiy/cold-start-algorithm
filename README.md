# Cold-Start Recommendation Algorithm

Research-grade recommendation system for cold-start scenarios.

## Quick Start

https://grouplens.org/datasets/serendipity-2018/

https://github.com/greenblue96/Taobao-Serendipity-Dataset

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Встроенно в функцию ниже, больше не актуальный шаг 

### 3. Run Experiments

главная команда для быстрого теста всех метрик и тд и тп, остальные команды для точечных тестов. 

перед использованием, обязательно создать пустую папку experiments в корне проекта в контексе где лежат data, src, tools. спустя время в папке experiments будет создан файл runs.jsonl - там данные о экспериментах 

```

python -m tools.full_pipeline --clean --fast --rebuild-gt
```


ниже команда просто запускает экспы, без метрик

```

python -m src.run_all_experiments --baselines-only
```

**Run all experiments (baselines + ablation)** — uses 500 test users and 5 seeds by default (see `src/evaluation_config.py`):
```bash
python -m src.run_all_experiments

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

## Experiment Configurations

### Baselines

1. **Random**: Random item selection
2. **Popularity**: Most popular items
3. **Embedding cosine**: Cosine similarity between user profile and items (no reranker)

### Ablation Study

1. **candidates_only**: FAISS retrieval only (no reranker)
2. **with_reranker**: FAISS + Cross-encoder reranker

# Project Structure

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

## Research Workflow

1. **Run experiments** → logged to `runs.jsonl`
2. **Analyze results** → load from `runs.jsonl`
3. **Generate tables/plots** → from unified log
4. **Answer reviewers** → all runs are logged automatically

## Notes

- **No simulation**: Uses real ground-truth data only
- **Cold-start**: User profiles are minimal (no history)
- **Reproducible**: All runs logged with seed and config
- **Scalable**: Supports 500+ users

## Cleanup

Remove demo/test files:
```bash
python tools/cleanup_demo_files.py
```

Dry-run first:
```bash
python tools/cleanup_demo_files.py --dry-run
```



python -m tools.build_master_results


python -m tools.advanced_plotting


python -m tools.enhanced_stat_tests


python -m tools.hypothesis_analysis


python -m tools.analyze_scores


python -m tools.error_analysis


python -m tools.fix_gt_catalog --replace


python -m tools.fix_gt_catalog



python -m tools.fix_gt_catalog


python -m src.run_all_experiments --n-users 500 --seeds 42 7 123


python -m src.run_ablation_pool_sizes --pool-sizes 200 500 1000


python -m tools.build_master_results
python -m tools.aggregate_runs


python -m tools.hypothesis_analysis
python -m tools.analyze_scores
python -m tools.error_analysis
python -m tools.enhanced_stat_tests


python -m tools.plotting
python -m tools.advanced_plotting

python -m tools.generate_paper_tables

