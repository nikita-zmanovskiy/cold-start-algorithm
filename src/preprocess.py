# preprocess.py
import os
import pandas as pd
from pathlib import Path
from .config import DATA_DIR, PROCESSED_DIR
from .utils import logger, save_json

def prepare_serendipity2018(src_dir: Path, out_dir: Path):
    """
    Try to find common Serendipity-2018 files and produce unified csvs:
    interactions.csv (user_id,item_id,timestamp,rating,serendipity)
    items.csv (item_id,title,genres,raw_text)
    users.csv (user_id,age,gender)
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing Serendipity-2018...")

    # heuristics for file names
    # look for 'serendipity' folder files
    # Attempt loading some typical files, otherwise fall back to scanning CSVs
    items = []
    interactions = []
    users = []

    # Try movieLens style files
    possible_movies = list(src_dir.glob("**/movies*.csv")) + list(src_dir.glob("**/movies*.tsv"))
    possible_ratings = list(src_dir.glob("**/ratings*.csv")) + list(src_dir.glob("**/ratings*.tsv"))
    if possible_movies:
        # movies = pd.read_csv(possible_movies[0], sep=None, engine='python')
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
        # ensure id and title
        if 'movieId' in movies.columns:
            movies = movies[['movieId','title','genres']].rename(columns={'movieId':'item_id'})
        movies.to_csv(out_dir / "items_serendipity.csv", index=False)
        logger.info(f"Wrote items_serendipity.csv, shape={movies.shape}")
    else:
        logger.warning("movies file not found for Serendipity dataset; check your data folder")

    if possible_ratings:
        ratings = pd.read_csv(possible_ratings[0], sep=None, engine='python')
        if 'userId' in ratings.columns:
            ratings = ratings.rename(columns={'userId':'user_id','movieId':'item_id','rating':'rating'})
        ratings.to_csv(out_dir / "interactions_serendipity.csv", index=False)
        logger.info(f"Wrote interactions_serendipity.csv, shape={ratings.shape}")
    else:
        logger.warning("ratings file not found for Serendipity dataset; check your data folder")

def prepare_taobao(src_dir: Path, out_dir: Path):
    """
    Simplified loader for Taobao Serendipity dataset. The real dataset may have custom format.
    Attempt to find files and convert them to unified CSVs.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing Taobao-Serendipity...")

    # Heuristic: find csv/tsv in folder
    csvs = list(src_dir.glob("**/*.csv")) + list(src_dir.glob("**/*.tsv"))
    if not csvs:
        logger.warning("No CSV/TSV found in Taobao folder; please inspect dataset")
        return
    # try to find user/item interactions by column names
    for f in csvs:
        try:
            df = pd.read_csv(f, sep=None, engine='python', low_memory=False)
            cols = [c.lower() for c in df.columns]
            if any('user' in c for c in cols) and any('item' in c or 'goods' in c for c in cols):
                df.to_csv(out_dir / "interactions_taobao.csv", index=False)
                logger.info(f"Found interactions-like file: {f.name} -> interactions_taobao.csv")
                break
        except Exception:
            continue

def run_all():
    # call both loaders with expected folders
    seren_path = DATA_DIR / "serendipity-sac2018"
    taobao_path = DATA_DIR / "Taobao-Serendipity-Dataset-master"
    prepare_serendipity2018(seren_path, PROCESSED_DIR)
    prepare_taobao(taobao_path, PROCESSED_DIR)
    logger.info("Preprocessing finished. Check data/processed/ for outputs.")

if __name__ == "__main__":
    run_all()
