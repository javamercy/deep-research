from deep_research.configuration import LLMModel, LLMModelConfig
from evals.scoping.dataset import BRIEF_DATASET_NAME, PLAN_DATASET_NAME
from evals.scoping.evaluators import create_research_brief_evaluators, create_research_plan_evaluators
from evals.scoping.target import brief_target, plan_target
from evals.spec import EvalSpec


def create_brief_spec() -> EvalSpec:
    judge_config = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
    )

    return EvalSpec(
        name="brief",
        description="Evaluate research brief coverage and groundedness.",
        dataset_name=BRIEF_DATASET_NAME,
        target=brief_target,
        evaluators=create_research_brief_evaluators(judge_config),
        experiment_prefix="brief",
    )


def create_plan_spec() -> EvalSpec:
    judge_config = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
    )

    return EvalSpec(
        name="plan",
        description="Evaluate research plan coverage and groundedness.",
        dataset_name=PLAN_DATASET_NAME,
        target=plan_target,
        evaluators=create_research_plan_evaluators(judge_config),
        experiment_prefix="plan",
    )
