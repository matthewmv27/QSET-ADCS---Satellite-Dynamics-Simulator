"""Run reproducible M2 configuration sweeps from a JSON input file."""
import argparse
from pathlib import Path

from edtsim.inertia.scenario import load_scenario
from edtsim.inertia.sweeps import plot_results, run_sweeps, save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="Versioned M2 scenario JSON file")
    parser.add_argument("--output-dir", "-o", required=True, type=Path, help="Folder for CSV and JSON output")
    parser.add_argument("--max-cases", type=int, default=2000, help="Safety limit (default 2000 cases)")
    parser.add_argument("--plot", action="store_true", help="Save optional PNG plots (requires .[viz])")
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    cases = run_sweeps(scenario, args.max_cases)
    jsonpath, csvpath = save_results(scenario, cases, args.output_dir)
    print("Evaluated {} cases for {} ({}).".format(len(cases), scenario.identifier, scenario.provenance["status"]))
    print("Saved {}".format(jsonpath))
    print("Saved {}".format(csvpath))
    if args.plot:
        for path in plot_results(cases, args.output_dir):
            print("Saved {}".format(path))
    print("Static mass-property snapshots only; no drag or transient dynamics.")


if __name__ == "__main__":
    main()
