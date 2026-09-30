import asyncio

from dotenv import load_dotenv
from langsmith import Client

from evals.research.dataset import ensure_research_dataset
from evals.research.evaluators import next_step_evaluator
from evals.research.target import research_target
from evals.scoping.dataset import ensure_scoping_dataset
from evals.scoping.evaluators import brief_criteria_evaluator, brief_groundedness_evaluator
from evals.scoping.target import scoping_target
from evals.supervisor.dataset import ensure_supervisor_dataset
from evals.supervisor.evaluators import supervisor_parallelization_evaluator
from evals.supervisor.target import supervisor_target


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


async def run_supervisor(langsmith_client: Client) -> None:
    dataset_name = "deep_research_supervisor"
    ensure_supervisor_dataset(langsmith_client=langsmith_client, dataset_name=dataset_name)

    results = await langsmith_client.aevaluate(
        supervisor_target,
        data=dataset_name,
        # pyrefly: ignore [bad-argument-type]
        evaluators=[supervisor_parallelization_evaluator],
        experiment_prefix="supervisor"
    )

    print(results)


async def main():
    load_dotenv()
    langsmith_client = Client()
    # await run_scoping(langsmith_client)
    # await run_research(langsmith_client)
    await run_supervisor(langsmith_client)


if __name__ == "__main__":
    asyncio.run(main())
