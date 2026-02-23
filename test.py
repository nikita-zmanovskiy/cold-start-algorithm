import os
from pathlib import Path

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def is_movielens_path(p: Path) -> bool:
    name = p.name.lower()
    # zip or extracted folder
    if name.startswith("ml-") and ("movielens" in name or name.endswith(".zip") or p.is_dir()):
        return True
    # key files
    if name in {"ratings.csv", "movies.csv", "tags.csv", "links.csv", "genome-scores.csv", "genome-tags.csv"}:
        return True
    # common extracted folder names like ml-25m, ml-20m
    if p.is_dir() and p.name.lower().startswith("ml-") and p.name.lower().endswith("m"):
        return True
    return False

def print_tree(root: Path, max_depth: int = 5, max_files_per_dir: int = 200, show_hidden: bool = False):
    root = root.resolve()
    if not root.exists():
        print(f"[!] Folder not found: {root}")
        return

    print(f"\n=== TREE: {root} ===\n")

    def _walk(dir_path: Path, depth: int):
        if depth > max_depth:
            print("  " * depth + "… (depth limit)")
            return

        try:
            entries = list(dir_path.iterdir())
        except PermissionError:
            print("  " * depth + f"[no permission] {dir_path.name}/")
            return

        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        # Sort: dirs first, then files
        entries.sort(key=lambda x: (x.is_file(), x.name.lower()))

        # Trim very large dirs
        trimmed = False
        if len(entries) > max_files_per_dir:
            entries = entries[:max_files_per_dir]
            trimmed = True

        for e in entries:
            marker = "🎬" if is_movielens_path(e) else "  "
            if e.is_dir():
                print("  " * depth + f"{marker} {e.name}/")
                _walk(e, depth + 1)
            else:
                try:
                    size = human_size(e.stat().st_size)
                except OSError:
                    size = "?"
                print("  " * depth + f"{marker} {e.name}  [{size}]")

        if trimmed:
            print("  " * depth + f"… ({dir_path.name} truncated; showing first {max_files_per_dir} entries)")

    _walk(root, 0)
    print("\n=== END ===\n")

def find_movielens(root: Path):
    print(f"\n=== MovieLens scan under: {root.resolve()} ===")
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and is_movielens_path(p):
            hits.append(p)
        elif p.is_dir() and p.name.lower().startswith("ml-") and p.name.lower().endswith("m"):
            hits.append(p)

    # de-dup + sort
    hits = sorted(set(hits), key=lambda x: str(x).lower())

    if not hits:
        print("[!] No MovieLens-looking files/folders found.")
        return

    for p in hits:
        if p.is_dir():
            print(f"🎬 DIR  {p}")
        else:
            try:
                size = human_size(p.stat().st_size)
            except OSError:
                size = "?"
            print(f"🎬 FILE {p}  [{size}]")
    print("=== END scan ===\n")

if __name__ == "__main__":
    # Adjust these if you want:
    data_root = Path("data")  # usually project_root/data
    print_tree(data_root, max_depth=6, max_files_per_dir=200, show_hidden=False)
    find_movielens(data_root)