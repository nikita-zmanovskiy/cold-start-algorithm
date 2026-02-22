
from pathlib import Path

N_TEST_USERS = 500

EVAL_SEEDS = [42, 7, 123, 2024, 2025, 0, 13, 999, 1234, 777]

N_BOOTSTRAP = 1000

N_QUICK_TEST_USERS = 50
N_SANITY_USERS = 5

N_FAST_USERS = 30
FAST_SEEDS = EVAL_SEEDS[:5]  
FAST_POOL_SIZES = [200, 500]

ABLATION_POOL_SIZES = [100, 300, 1000, 5000]

ABLATION_RETRIEVAL_MODES = ["ann", "bm25", "hybrid"]


SPLIT_METHOD = "time_based"  


SPLIT_RATIOS = (0.7, 0.1, 0.2)  


COLD_START_SCENARIOS = ("new_users", "new_items", "both")


NEW_USER_MAX_TRAIN = 0


INTERACTION_BUCKETS = [
    (1, 2),    
    (3, 5),    
    (6, 10),   
    (11, 20),  
    (21, None) 
]


NEGATIVE_SAMPLING_PROTOCOL = "full_ranking"
N_NEGATIVES_PER_POSITIVE = 100  
NEGATIVE_SAMPLING_SEED = 42

GT_PATH = Path(__file__).resolve().parents[1] / "experiments" / "ground_truth.json"
SPLIT_METADATA_PATH = Path(__file__).resolve().parents[1] / "experiments" / "split_metadata.json"
TRAINING_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "serendipity-sac2018" / "training.csv"
TEST_CSV_PATH = Path(__file__).resolve().parents[1] / "experiments" / "test_interactions.csv"
VAL_CSV_PATH = Path(__file__).resolve().parents[1] / "experiments" / "val_interactions.csv"
