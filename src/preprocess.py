
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
        logger.warning("Taobao dataset folder not found (%s); skipping.", src_dir)
        return
    logger.info("Preparing Taobao-Serendipity...")

    csvs = list(src_dir.glob("**/*.csv")) + list(src_dir.glob("**/*.tsv"))
    if not csvs:
        if not (out_dir / "interactions_taobao.csv").exists():
            logger.warning("No CSV/TSV in Taobao folder (%s); skipping.", src_dir)
        return

    written = False
    for f in csvs:
        try:
            try:
                df = pd.read_csv(f, sep=None, engine="python", dtype=str)
            except Exception:
                df = pd.read_csv(f, sep=",", engine="c", low_memory=False)  # fallback: обычный csv

            def _pick_col(cols_lower, candidates):
                for cand in candidates:
                    for i, c in enumerate(cols_lower):
                        if c == cand or c.endswith(cand) or cand in c:
                            return i
                return None

            cols = list(df.columns)
            cols_lower = [str(c).lower() for c in cols]

            # --- pick user/item columns (supports UserId/ItemId etc.) ---
            u_idx = _pick_col(cols_lower, ["user_id", "userid", "user", "uid", "userId".lower()])
            i_idx = _pick_col(cols_lower, ["item_id", "itemid", "item", "goods_id", "product_id", "movieid", "itemId".lower()])
            t_idx = _pick_col(cols_lower, ["timestamp", "time", "ts", "datetime"])

            if u_idx is None or i_idx is None:
                # not an interactions file; try next
                logger.warning("Skipping %s: can't infer user/item columns from %s", f.name, cols)
                continue

            u_col = cols[u_idx]
            i_col = cols[i_idx]

            # --- keep only needed cols ---
            keep = [u_col, i_col] + ([cols[t_idx]] if t_idx is not None else [])
            out = df[keep].copy()

            out = out.rename(columns={u_col: "user_id", i_col: "item_id"})
            if t_idx is not None:
                out = out.rename(columns={cols[t_idx]: "timestamp"})
            else:
                out["timestamp"] = ""

            # --- optional: if Clicked exists, keep only positives ---
            # Taobao-Serendipity dataset often uses 'Clicked' as implicit feedback
            clicked_col = None
            for c in df.columns:
                if str(c).lower() == "clicked":
                    clicked_col = c
                    break

            if clicked_col is not None:
                try:
                    clicked = pd.to_numeric(df[clicked_col], errors="coerce").fillna(0)
                    out = out.loc[clicked == 1].copy()
                except Exception:
                    logger.warning("Found Clicked column in %s but couldn't filter; keeping all rows.", f.name)

            # --- normalize types ---
            out["user_id"] = out["user_id"].astype(str)
            out["item_id"] = out["item_id"].astype(str)

            # --- write interactions ---
            interactions_path = out_dir / "interactions_taobao.csv"
            out[["user_id", "item_id", "timestamp"]].to_csv(interactions_path, index=False)

            # --- write items (minimal metadata) ---
            items_path = out_dir / "items_taobao.csv"
            items_df = (
                out[["item_id"]]
                .drop_duplicates()
                .assign(title=lambda x: "Item " + x["item_id"].astype(str))
            )
            items_df.to_csv(items_path, index=False)

            logger.info(
                "Taobao: wrote %s (rows=%d) and %s (items=%d) from %s",
                interactions_path.name, len(out),
                items_path.name, len(items_df),
                f.name,
            )

            written = True
            break

        except Exception as e:
            logger.warning("Failed to parse %s: %s", f.name, e)
            continue

    if not written and not (out_dir / "interactions_taobao.csv").exists():
        logger.warning("No interactions-like CSV/TSV in Taobao folder (%s); skipping.", src_dir)
def run_all():

    seren_path = DATA_DIR / "serendipity-sac2018"
    taobao_path = DATA_DIR / "Taobao-Serendipity-Dataset-master"
    prepare_serendipity2018(seren_path, PROCESSED_DIR)
    prepare_taobao(taobao_path, PROCESSED_DIR)
    logger.info("Preprocessing finished. Check data/processed/ for outputs.")

if __name__ == "__main__":
    run_all()
