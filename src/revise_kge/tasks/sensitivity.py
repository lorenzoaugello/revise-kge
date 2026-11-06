from revise_kge.engine import P, run_experiment, tune_all

def main():
    """Run sensitivity analyses for contraction, locality, and anchoring."""
    tune_all(P.sens)
    run_experiment("sens", P.sens, ["mixed"], P.seeds_aux, "sens")

if __name__ == "__main__":
    main()
