from evals.research.dataset import DATASET_NAME
from evals.research.evaluators import create_research_evaluators
from evals.research.target import research_target
from evals.spec import EvalSpec


def create_research_spec() -> EvalSpec:
    return EvalSpec(
        name="research",
        description="Evaluate the research agent's ability to conduct research and provide accurate, relevant information.",
        dataset_name=DATASET_NAME,
        evaluators=create_research_evaluators(),
        target=research_target,
        experiment_prefix="research"
    )
