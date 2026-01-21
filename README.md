# TinyBench Dataset Structure

This project generates QA pairs from datasets with injected data phenomena. I also added results from plugging questions into gpt5.2 with the perturbed dataset.

## JSON Files Overview

### Result Files (Useful stuff for evaluation)

Generated instances are stored in `data/instances/{dataset_name}/seed_{seed}/` directory. The primary outputs for evaluation are:

#### [`questions_and_answers.json`](data/instances/bike_sharing_10k/seed_42/questions_and_answers.json)
**Generated QA pairs** - the core evaluation dataset:
- `questions`: Array of question strings about data quality phenomena
- `answers`: Array of corresponding reference answers

This file are the questions extracted from what is automatically generated after injecting phenomena.

#### [`gpt-5.2_answers.json`](data/instances/bike_sharing_10k/seed_42/gpt-5.2_answers.json)
**LLM responses** to the generated questions:
- Just plugged questions into GPT5.2
- Shows how well GPT-5.2 performs when presented with the perturbed dataset
- Primary output for measuring LLM evaluation on data quality tasks

So far, GPT-5.2 got 100% on the 9 questions.

### Configuration Files

#### [`configs/datasets/bike_sharing_toy.json`](configs/datasets/bike_sharing_toy.json)
Defines the dataset configuration and data quality injections to apply:
- `dataset_name`: Name of the dataset instance
- `base_dataset`: Path to the source CSV file
- `summary`: Path to dataset summary statistics
- `injectors`: Array of data quality phenomena to inject (e.g., missing values, duplicate features, correlations)

### Supporting Files

#### [`manifest.json`](data/instances/bike_sharing_10k/seed_42/manifest.json)
Core metadata for a dataset instance:
- `dataset_instance_id`: Unique identifier for this instance
- `seed`: Random seed used for reproducibility
- `base_dataset`: Reference to source data
- `phenomena`: Array of applied injections with:
  - `type`: Injection type (e.g., "mcar_missingness")
  - `params`: Injection parameters
  - `effects`: Measured impact (e.g., number of nulled cells)
  - `questions`: QA pairs generated for this phenomenon

#### [`instance_summary.json`](data/instances/bike_sharing_10k/seed_42/instance_summary.json)
Statistical summary of the dataset instance:
- `n_rows`: Number of rows
- `n_cols`: Number of columns
- `columns`: List of column names
- `null_fraction_by_col`: Missing value fraction per column

#### `table.csv`
The actual dataset instance with all injected phenomena applied.

## Workflow

1. **Configuration** → [`configs/datasets/bike_sharing_toy.json`](configs/datasets/bike_sharing_toy.json)
2. **Generation** → [`src/generate_instance.py`](src/generate_instance.py) creates instance files
3. **QA Extraction** → [`get_qa_pairs.ipynb`](get_qa_pairs.ipynb) generates [`questions_and_answers.json`](data/instances/bike_sharing_10k/seed_42/questions_and_answers.json)
4. **LLM Evaluation** → Responses saved to `gpt-5.2_answers.json`