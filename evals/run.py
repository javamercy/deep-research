import asyncio

from dotenv import load_dotenv
from langsmith import Client

from evals.scoping.dataset import ensure_scoping_dataset
from evals.scoping.evaluators import brief_criteria_evaluator, brief_groundedness_evaluator
from evals.scoping.target import scoping_target


async def run_scoping() -> None:
    langsmith_client = Client()

    dataset_name = "deep_research_scoping"
    ensure_scoping_dataset(langsmith_client=langsmith_client, dataset_name=dataset_name)

    results = await langsmith_client.aevaluate(
        scoping_target,
        data=dataset_name,
        # pyrefly: ignore [bad-argument-type]
        evaluators=[brief_criteria_evaluator, brief_groundedness_evaluator],
        experiment_prefix="scoping"
    )

    print(results)


async def main():
    load_dotenv()
    await run_scoping()


if __name__ == "__main__":
    asyncio.run(main())
