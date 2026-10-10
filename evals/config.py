from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langsmith import get_current_run_tree


def create_eval_config(**values) -> RunnableConfig:
    thread_id = str(uuid4())

    if run_tree := get_current_run_tree():
        run_tree.add_metadata({"thread_id": thread_id})

    return RunnableConfig(configurable={"thread_id": thread_id, **values})
