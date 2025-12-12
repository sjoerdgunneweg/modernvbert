
from datasets import load_dataset
import random
import numpy as np
import pandas as pd

SAMPLE_SIZE = 1000
MIN_LEN = 30
MAX_LEN = 90

def extract_queries(dataset_name: str) -> list:
    """Loads a dataset and extracts the 'query' column from the 'test' split."""
    try:
        dataset = load_dataset(dataset_name, split='test')
        
        # Access the 'query' column. The result is a list-like object.
        return dataset['query']
        
    except Exception as e:
        # Handle cases where the split or column name is different
        print(f"Error loading {dataset_name} split 'test' or column 'query': {e}")
        # Try a common alternative split/column combination if needed, or raise.
        return []

def save_queries_to_csv(queries: list, filename: str) -> None:
    """Saves a list of queries to a CSV file."""
    df_queries = pd.DataFrame(queries, columns=['query'])
    df_queries.to_csv(filename, index=False)
    print(f"Saved {len(queries)} queries to {filename}")

def main():
    dataset_name_1 = "vidore/arxivqa_test_subsampled"
    dataset_name_2 = "vidore/esg_reports_v2"
    dataset_name_3 = "vidore/docvqa_test_subsampled"
    dataset_name_4 = "vidore/infovqa_test_subsampled"
    dataset_name_5 = "vidore/tatdqa_test"

    queries_1 = extract_queries(dataset_name_1)
    queries_2 = extract_queries(dataset_name_2)
    queries_3 = extract_queries(dataset_name_3)
    queries_4 = extract_queries(dataset_name_4)
    queries_5 = extract_queries(dataset_name_5)


    all_queries = list(queries_1)

    all_queries.extend(queries_2)
    all_queries.extend(queries_3)
    all_queries.extend(queries_4)
    all_queries.extend(queries_5)



    print(f"Loaded {len(all_queries)} total queries.")
    print(f"Average length of combined queries: {np.mean([len(q) for q in all_queries]):.2f} characters.")


    filtered_queries = [q for q in all_queries if MIN_LEN <= len(q) <= MAX_LEN]

    print(f"Filtered to {len(filtered_queries)} queries between {MIN_LEN} and {MAX_LEN} characters.")

    if len(filtered_queries) < SAMPLE_SIZE:
        print("Warning: Not enough filtered queries. Using all available filtered queries.")
        final_queries = filtered_queries
    else:
        final_queries = random.sample(filtered_queries, SAMPLE_SIZE)
    average_length = np.mean([len(q) for q in final_queries])

    print(f"\nFinal Sample Size: {len(final_queries)}")
    print(f"Average Query Length: {average_length:.2f} characters.")
    print(f"Target Average Length: 60 characters (as per paper).")


    save_queries_to_csv(final_queries, "sampled_queries.csv")

    

if __name__ == "__main__":
    main()