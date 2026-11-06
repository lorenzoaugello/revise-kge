from revise_kge.engine import P, run_experiment, tune_all

def main():
    """Run component ablations on the configured reduced dataset/model grid."""
    tune_all(P.abl)
    run_experiment("abl", P.abl, ["mixed"], P.seeds, "abl")

if __name__ == "__main__":
    main()
