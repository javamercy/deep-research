from deep_research.configuration import LLMModel, LLMModelConfig
from evals.scoping.dataset import DATASET_NAME
from evals.scoping.evaluators import create_research_brief_evaluators
from evals.scoping.target import research_brief_target
from evals.spec import EvalSpec


def creating_scoping_spec() -> EvalSpec:
    judge_config = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
        temperature=0,
    )

    return EvalSpec(
        name="scoping",
        description="Evaluate research brief coverage and groundedness.",
        dataset_name=DATASET_NAME,
        target=research_brief_target,
        evaluators=create_research_brief_evaluators(judge_config),
        experiment_prefix="scoping",
    )
