import os
from pathlib import Path

def print_tree_with_tif_samples(root_dir: str, max_tif_per_dir: int = 2):
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")

    def walk(dir_path: Path, prefix: str = ""):
        entries = sorted([p for p in dir_path.iterdir() if not p.name.startswith(".")],
                         key=lambda p: (p.is_file(), p.name.lower()))

        dirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file()]

        tif_files = [p for p in files if p.suffix.lower() == ".tif"]
        other_files = [p for p in files if p.suffix.lower() != ".tif"]

        if tif_files:
            sample = tif_files[:max_tif_per_dir]
            print(f"{prefix}{dir_path.name}/  ({len(tif_files)} .tif)")
            for i, f in enumerate(sample, 1):
                print(f"{prefix}  - {f.name}")
            if other_files:
                print(f"{prefix}  ... (+{len(other_files)} non-tif files)")
        else:
            print(f"{prefix}{dir_path.name}/")
            for f in other_files[:2]:
                print(f"{prefix}  - {f.name}")
            if len(other_files) > 2:
                print(f"{prefix}  ... (+{len(other_files)-2} more files)")

        for d in dirs:
            walk(d, prefix + "  ")

    print(str(root))
    walk(root, "")

if __name__ == "__main__":
    print_tree_with_tif_samples("data/COLUMBIA", max_tif_per_dir=2)
