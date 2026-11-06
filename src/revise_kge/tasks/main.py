from revise_kge.engine import P, run_experiment, tune_all

def main():
    """Run the mixed addition, deletion, and substitution evaluation."""
    tune_all(P.main)
    run_experiment("main", P.main, ["mixed"], P.seeds, "main")

if __name__ == "__main__":
    main()
