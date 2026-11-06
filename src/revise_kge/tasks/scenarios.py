from revise_kge.engine import P, run_experiment, tune_all

def main():
    """Run removal-only, substitution-only, and entity-deletion scenarios."""
    tune_all(P.scen)
    run_experiment("scen", P.scen, ["remove", "substitute", "entity_deletion"], P.seeds_aux, "main")

if __name__ == "__main__":
    main()
