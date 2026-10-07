from pathlib import Path

from langsmith import Client
from langsmith.schemas import Dataset

from evals.datasets import ensure_dataset, load_examples_from_jsonl

DATASET_NAME = "deep_research_supervisor"
EXAMPLES_PATH = Path(__file__).parent / "data" / "examples.jsonl"


def ensure_supervisor_dataset(ls_client: Client) -> Dataset:
    examples = load_examples_from_jsonl(EXAMPLES_PATH)
    return ensure_dataset(
        ls_client,
        dataset_name=DATASET_NAME,
        examples=examples
    )
