import argparse
import json

ViDoRe_V1_TASKS =[
            "VidoreArxivQARetrieval",
            "VidoreDocVQARetrieval",
            "VidoreInfoVQARetrieval",
            "VidoreTabfquadRetrieval",
            "VidoreTatdqaRetrieval",
            "VidoreShiftProjectRetrieval",
            "VidoreSyntheticDocQAAIRetrieval",
            "VidoreSyntheticDocQAEnergyRetrieval",
            "VidoreSyntheticDocQAGovernmentReportsRetrieval",
            "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        ]

ViDoRe_V2_TASKS = [
            "Vidore2ESGReportsRetrieval",
            "Vidore2EconomicsReportsRetrieval",
            "Vidore2BioMedicalLecturesRetrieval",
            "Vidore2ESGReportsHLRetrieval",
        ]

ViDoRe_V3_TASKS = [
            "Vidore3FinanceEnRetrieval",
            "Vidore3IndustrialRetrieval",
            "Vidore3ComputerScienceRetrieval",
            "Vidore3PharmaceuticalsRetrieval",
            "Vidore3HrRetrieval",
            "Vidore3FinanceFrRetrieval",
            "Vidore3PhysicsRetrieval",
            "Vidore3EnergyRetrieval",
            # "Vidore3TelecomRetrieval", # NOTE: these two datasets are disabled since no acces to these private datasets
            # "Vidore3NuclearRetrieval",
        ]

def json_reader(filepath, task_name):
    with open(filepath, 'r') as f:
        data = json.load(f)
        print(f"Reading {task_name}: {data['scores']['test'][0]['ndcg_at_5']}")
    return data["scores"]["test"][0]["ndcg_at_5"]

def get_model_name_from_path(filepath):
    # Assuming the filepath is something like "results/model_name/no_revision_available/task_name.json"
    filepath_parts = filepath.split('/')
    return filepath_parts[1]

def main(args):
    print("Aggregating nCDG@5 scores of model: " + get_model_name_from_path(args.filepath) + " over benchmark: " + args.benchmark)
    if args.benchmark == "ViDoRe_V1":
        task_list = ViDoRe_V1_TASKS
    elif args.benchmark == "ViDoRe_V2":
        task_list = ViDoRe_V2_TASKS
    elif args.benchmark == "ViDoRe_V3":
        task_list = ViDoRe_V3_TASKS
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    summed_ncdg_at_5 = sum([json_reader(f"{args.filepath}/{task}.json", task) for task in task_list])
    aggregated_ncdg_at_5 = summed_ncdg_at_5 / len(task_list)
    print(f"Aggregated nCDG@5 for {args.benchmark}: {aggregated_ncdg_at_5}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--filepath", "-f",
        type=str,
        required=True,
        help="Path to the folder containing the JSON result files for each task.",
    )
    args.add_argument(
        "--benchmark", "-b",
        type=str,
        required=True,
        choices=["ViDoRe_V1", "ViDoRe_V2", "ViDoRe_V3"],
        help="List of tasks to aggregate nCDG@5 over.",
    )
    args = args.parse_args()

    main(args)