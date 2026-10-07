from pathlib import Path
from uuid import uuid5

from langsmith import Client
from langsmith.schemas import Dataset
from pydantic import BaseModel, Field


class DatasetExample(BaseModel):
    id: str
    inputs: dict
    outputs: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


def load_examples_from_jsonl(path: Path) -> list[DatasetExample]:
    examples = [
        DatasetExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example IDs found in {path}")

    return examples


def ensure_dataset(
        ls_client: Client,
        *,
        dataset_name: str,
        examples: list[DatasetExample]
) -> Dataset:
    dataset = (
        ls_client.read_dataset(dataset_name=dataset_name)
        if ls_client.has_dataset(dataset_name=dataset_name)
        else ls_client.create_dataset(dataset_name=dataset_name)
    )

    existing_examples = {
        example.id: example
        for example in ls_client.list_examples(dataset_id=dataset.id)
    }

    additions: list[dict] = []
    for example in examples:
        example_id = uuid5(dataset.id, example.id)
        payload = example.model_dump(exclude={"id"})
        current_example = existing_examples.get(example_id)

        if current_example is None:
            additions.append({"id": example_id, **payload})
        elif (
                current_example.inputs != example.inputs
                or (current_example.outputs or {}) != example.outputs
                or (current_example.metadata or {}) != example.metadata
        ):
            ls_client.update_example(example_id=example_id, **payload)

    if additions:
        ls_client.create_examples(dataset_id=dataset.id, examples=additions)

    return dataset
