from pathlib import Path

from langsmith import Client
from langsmith.schemas import Dataset

from evals.datasets import ensure_dataset, load_examples_from_jsonl

BRIEF_DATASET_NAME = "deep_research_brief"
BRIEF_EXAMPLES_PATH = Path(__file__).parent / "data" / "brief_examples.jsonl"

PLAN_DATASET_NAME = "deep_research_plan"
PLAN_EXAMPLES_PATH = Path(__file__).parent / "data" / "plan_examples.jsonl"


def ensure_brief_dataset(ls_client: Client) -> Dataset:
    examples = load_examples_from_jsonl(BRIEF_EXAMPLES_PATH)
    return ensure_dataset(
        ls_client=ls_client,
        dataset_name=BRIEF_DATASET_NAME,
        examples=examples
    )


def ensure_plan_dataset(ls_client: Client) -> Dataset:
    examples = load_examples_from_jsonl(PLAN_EXAMPLES_PATH)
    return ensure_dataset(
        ls_client=ls_client,
        dataset_name=PLAN_DATASET_NAME,
        examples=examples
    )
