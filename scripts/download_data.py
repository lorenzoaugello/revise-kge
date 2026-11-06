"""Download and validate the four public benchmark splits."""
from hashlib import sha256
from revise_kge.engine import DATA_DIR, load_kg

if __name__ == "__main__":
    for dataset in ("FB15k-237", "WN18RR", "CoDEx-S", "CoDEx-M"):
        graph = load_kg(dataset, max_entities=None)
        print(f"{dataset}: entities={graph['n_ent']:,}, relations={graph['n_rel']:,}, "
              f"train={len(graph['train']):,}, valid={len(graph['valid']):,}, test={len(graph['test']):,}")
    files = sorted(DATA_DIR.glob("*/*.txt"))
    manifest = DATA_DIR / "SHA256SUMS.txt"
    manifest.write_text("".join(f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(DATA_DIR)}\n" for path in files))
    print(f"Raw split files are cached under {DATA_DIR}; checksums: {manifest}")
