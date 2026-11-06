"""Run the full experiment sequence in paper configuration."""
from revise_kge.tasks.tune import main as tune
from revise_kge.tasks.main import main as run_main
from revise_kge.tasks.scenarios import main as run_scenarios
from revise_kge.tasks.ablations import main as run_ablations
from revise_kge.tasks.sensitivity import main as run_sensitivity

if __name__ == "__main__":
    tune()
    run_main()
    run_scenarios()
    run_ablations()
    run_sensitivity()
