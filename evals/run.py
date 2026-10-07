import argparse
import asyncio

from dotenv import load_dotenv
from langsmith import Client

from evals.research.suite import create_research_spec
from evals.scoping.suite import creating_scoping_spec
from evals.supervisor.suite import create_supervisor_spec

SUITES = {
    "scoping": creating_scoping_spec,
    "research": create_research_spec,
    "supervisor": create_supervisor_spec
}


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run evaluation suites.")
    parser.add_argument(
        "suite",
        type=str,
        choices=sorted(SUITES.keys()),
        help="Name of the evaluation suite to run."
    )
    args = parser.parse_args()

    spec = SUITES[args.suite]()

    await Client().aevaluate(
        spec.target,
        data=spec.dataset_name,
        evaluators=spec.evaluators,
        experiment_prefix=spec.experiment_prefix,
        description=spec.description
    )


if __name__ == "__main__":
    asyncio.run(main())
