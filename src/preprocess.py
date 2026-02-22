
import os
import pandas as pd
from pathlib import Path
from .config import DATA_DIR, PROCESSED_DIR
from .utils import logger, save_json

def prepare_serendipity2018(src_dir: Path, out_dir: Path):

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing Serendipity-2018...")


    items = []
    interactions = []
    users = []

    possible_movies = list(src_dir.glob("**/movies*.csv")) + list(src_dir.glob("**/movies*.tsv"))
    # Serendipity-SAC2018 uses training.csv (userId, movieId, rating); also check ratings*.csv
    possible_ratings = (
        list(src_dir.glob("**/ratings*.csv")) + list(src_dir.glob("**/ratings*.tsv")) +
        list(src_dir.glob("**/training*.csv")) + list(src_dir.glob("**/training*.tsv"))
    )
    if possible_movies:
        movies = pd.read_csv(
            possible_movies[0],
            sep=",",
            engine="python",
            quotechar='"',
            escapechar="\\",
            on_bad_lines="skip",
            encoding="utf-8",
        )
        movies = movies.rename(columns={c: c.strip() for c in movies.columns})
        if 'movieId' in movies.columns:
            movies = movies[['movieId','title','genres']].rename(columns={'movieId':'item_id'})
        movies.to_csv(out_dir / "items_serendipity.csv", index=False)
        logger.info(f"Wrote items_serendipity.csv, shape={movies.shape}")
    else:
        if not (out_dir / "items_serendipity.csv").exists():
            logger.warning("movies file not found for Serendipity dataset; check your data folder")
        else:
            logger.debug("movies file not found; using existing items_serendipity.csv")

    if possible_ratings:
        ratings = pd.read_csv(possible_ratings[0], sep=None, engine='python')
        if 'userId' in ratings.columns:
            ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id', 'rating': 'rating'})
        ratings.to_csv(out_dir / "interactions_serendipity.csv", index=False)
        logger.info(f"Wrote interactions_serendipity.csv, shape={ratings.shape}")
    else:
        if not (out_dir / "interactions_serendipity.csv").exists():
            logger.warning("ratings/training file not found for Serendipity dataset; check your data folder")
        else:
            logger.debug("ratings/training file not found; using existing interactions_serendipity.csv")

def prepare_taobao(src_dir: Path, out_dir: Path):

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        logger.debug("Taobao dataset folder not found (%s); skipping.", src_dir)
        return
    logger.info("Preparing Taobao-Serendipity...")

    csvs = list(src_dir.glob("**/*.csv")) + list(src_dir.glob("**/*.tsv"))
    if not csvs:
        if not (out_dir / "interactions_taobao.csv").exists():
            logger.debug("No CSV/TSV in Taobao folder (%s); skipping.", src_dir)
        return

    written = False
    for f in csvs:
        try:
            df = pd.read_csv(f, sep=None, engine='python', low_memory=False)
            cols = [c.lower() for c in df.columns]
            if any('user' in c for c in cols) and any('item' in c or 'goods' in c for c in cols):
                df.to_csv(out_dir / "interactions_taobao.csv", index=False)
                logger.info(f"Found interactions-like file: {f.name} -> interactions_taobao.csv")
                written = True
                break
        except Exception:
            continue
    if not written and not (out_dir / "interactions_taobao.csv").exists():
        logger.debug("No interactions-like CSV/TSV in Taobao folder (%s); skipping.", src_dir)

def run_all():

    seren_path = DATA_DIR / "serendipity-sac2018"
    taobao_path = DATA_DIR / "Taobao-Serendipity-Dataset-master"
    prepare_serendipity2018(seren_path, PROCESSED_DIR)
    prepare_taobao(taobao_path, PROCESSED_DIR)
    logger.info("Preprocessing finished. Check data/processed/ for outputs.")

if __name__ == "__main__":
    run_all()
