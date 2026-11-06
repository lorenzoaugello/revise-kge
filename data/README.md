# Data and provenance

The benchmark split files can be downloaded on demand by `scripts/download_data.py` and cached under `data/raw/`. The current source URLs are declared in `src/revise_kge/engine.py`. We do not redistribute the benchmark data directly in this repository, while we indicate to follow the original dataset and source-repository terms when downloading them.
Here are the references and links to their hugginface repositories:
- https://huggingface.co/datasets/KGraph/FB15k-237
- https://huggingface.co/datasets/VLyb/WN18RR
- https://github.com/tsafavi/codex

The evolving snapshots and additions, deletions, and substitutions are generated deterministically at run time from each benchmark's training split and the configured random seed. They are synthetic update streams, not external change logs. This is one limitation of our work that we state in the paper, and for which we outline a direction for future work. Validation triples are split deterministically into calibration and evaluation halves; test triples are reserved for final evaluation.

`results_snapshot/` contains the raw main-experiment CSV records bundled with this release. We provide those CSV files for transparency in the review process, and so to make the reported summaries inspectable without rerunning training. However, they are not required as inputs to the experiment scripts, while they serve as sole demonstration.
