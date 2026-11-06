from revise_kge.engine import P, tune_all

def main():
    """Select the contraction quantile on the independent validation stream."""
    tune_all(P.main)

if __name__ == "__main__":
    main()
