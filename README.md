````markdown
# Cold-Start Algorithm (Retrieval + LLM/Reranking Experiments)

https://files.grouplens.org/datasets/movielens/ml-25m.zip


> **One-command fast run (start here):**
```bash
python -m tools.full_pipeline --clean --fast --rebuild-gt
````

This repository provides an end-to-end pipeline for cold-start evaluation: preprocessing, dataset splits, ground-truth construction, retrieval/reranking experiments, and result aggregation.

---

## Requirements

* **Python 3.12.7** (used for development)
* Recommended: `venv` (or conda)

Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## What the fast pipeline does

Running:

```bash
python -m tools.full_pipeline --clean --fast --rebuild-gt
```

will:

1. Clean previously generated artifacts (when `--clean` is enabled)
2. Preprocess datasets and write normalized files to `data/processed/`
3. Create train/val/test splits + cold-start segments
4. Build / rebuild ground truth (when `--rebuild-gt` is enabled)
5. Run a small, quick experiment subset (`--fast`) to validate everything works end-to-end
6. Save logs and results into `experiments/`

---

## Data layout

### Serendipity-2018 (required)

Place the raw interactions CSV here:

```
data/serendipity-sac2018/training.csv
```

After preprocessing, the pipeline creates:

* `data/processed/items_serendipity.csv`
* `data/processed/interactions_serendipity.csv`

### Taobao-Serendipity (optional)

If you want to run Taobao experiments, place the dataset folder here:

```
data/Taobao-Serendipity-Dataset-master/
```

After preprocessing (if files are found and parsed), the pipeline creates:

* `data/processed/items_taobao.csv`
* `data/processed/interactions_taobao.csv`

If Taobao is not present, Serendipity still runs.

---

## Outputs

After running the pipeline, check:

### `experiments/`

Typical outputs:

* `training_interactions_*.csv`
* `val_interactions_*.csv`
* `test_interactions_*.csv`
* `ground_truth_*.json`
* `split_metadata_*.json`
* `runs.jsonl` (main experiment log)

### `data/processed/`

* normalized interactions and items CSVs used by the pipeline

---

## Cold-start scenarios

Splits support the following cold-start segments:

* `new_users`: users with zero train interactions
* `new_items`: interactions with items not seen in train
* `both`: new users interacting with new items

---

## Useful commands

### Preprocess only

```bash
python -m src.preprocess
```

### Build splits + ground truth

```bash
python -m src.create_splits --dataset serendipity --csv data/serendipity-sac2018/training.csv
# Optional (Taobao)
python -m src.create_splits --dataset taobao --csv data/processed/interactions_taobao.csv --random
```

### Run experiments directly

```bash
python -m src.run_all_experiments --dataset serendipity --n-users 500 --seeds 42
```

---

## Troubleshooting

### `FileNotFoundError: data/processed/interactions_taobao.csv`

Taobao dataset is missing or preprocessing did not generate the file.

* Ensure Taobao files exist under `data/Taobao-Serendipity-Dataset-master/`
* Or run Serendipity only.

### Installation issues on Windows (FAISS / Torch)

Some combinations of Python version + platform can cause install friction for heavy deps.
If needed:

* use a clean venv
* try a different compatible Python (3.10/3.11)
* or adjust experiment settings to avoid components that require problematic packages

---

## License

MIT License — see `LICENSE`.

```
::contentReference[oaicite:0]{index=0}
```
