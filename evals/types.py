from collections.abc import Awaitable, Callable

from langsmith import EvaluationResult

type TargetFunc = Callable[[dict], Awaitable[dict]]

type EvaluatorFunc = Callable[..., Awaitable[EvaluationResult]]
