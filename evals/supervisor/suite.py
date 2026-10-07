from evals.spec import EvalSpec
from evals.supervisor.dataset import DATASET_NAME
from evals.supervisor.evaluators import create_supervisor_evaluators
from evals.supervisor.target import supervisor_target


def create_supervisor_spec() -> EvalSpec:
    return EvalSpec(
        name="supervisor",
        description="Evaluate the supervisor's ability to manage and parallelize tasks.",
        dataset_name=DATASET_NAME,
        target=supervisor_target,
        evaluators=create_supervisor_evaluators(),
        experiment_prefix="supervisor"
    )
