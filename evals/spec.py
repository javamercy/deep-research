from deep_research.schemas import FrozenBaseModel
from evals.types import EvaluatorFunc, TargetFunc


class EvalSpec(FrozenBaseModel):
    name: str
    description: str
    dataset_name: str
    target: TargetFunc
    evaluators: list[EvaluatorFunc]
    experiment_prefix: str
