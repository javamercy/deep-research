import asyncio

from dotenv import load_dotenv
from langsmith import Client

from evals.research.dataset import ensure_research_dataset
from evals.research.evaluators import next_step_evaluator
from evals.research.target import research_target
from evals.scoping.dataset import ensure_scoping_dataset
from evals.scoping.evaluators import brief_criteria_evaluator, brief_groundedness_evaluator
from evals.scoping.target import scoping_target


async def run_scoping(langsmith_client: Client) -> None:
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


async def run_research(langsmith_client: Client) -> None:
    dataset_name = "deep_research_research"
    ensure_research_dataset(langsmith_client=langsmith_client, dataset_name=dataset_name)

    results = await langsmith_client.aevaluate(
        research_target,
        data=dataset_name,
        # pyrefly: ignore [bad-argument-type]
        evaluators=[next_step_evaluator],
        experiment_prefix="research"
    )

    print(results)


async def main():
    load_dotenv()
    langsmith_client = Client()
    # await run_scoping(langsmith_client)
    await run_research(langsmith_client)


if __name__ == "__main__":
    asyncio.run(main())
