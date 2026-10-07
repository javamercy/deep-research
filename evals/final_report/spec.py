from deep_research.configuration import LLMModel, LLMModelConfig
from evals.final_report.dataset import DATASET_NAME
from evals.final_report.evaluators import create_final_report_evaluators
from evals.final_report.target import final_report_target
from evals.spec import EvalSpec


def create_final_report_spec() -> EvalSpec:
    judge_config = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
        temperature=0.1,
        reasoning_effort="high"
    )

    return EvalSpec(
        name="final_report_evaluation",
        description="Evaluates how well a final report addresses a set of requirements.",
        dataset_name=DATASET_NAME,
        target=final_report_target,
        evaluators=create_final_report_evaluators(judge_config),
        experiment_prefix="final_report"
    )
