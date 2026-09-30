import asyncio
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from deep_research.state import AgentInputState, AgentState
from deep_research.utils import display_markdown, display_messages
from deep_research.workflows.deep_research import build_deep_research_graph


async def main():
    load_dotenv()

    config = RunnableConfig(
        configurable={"thread_id": uuid4()},
        recursion_limit=50,
    )
    graph = build_deep_research_graph(InMemorySaver())
    input_state = AgentInputState(
        messages=[HumanMessage(content="Let's compare GPT-6 Astra and Claude Opus 5.5")]
    )

    result = cast(AgentState, await graph.ainvoke(input_state, config=config))

    console = Console()
    display_messages(result["messages"], console=console)
    display_markdown(result["final_report"], console=console)


if __name__ == "__main__":
    asyncio.run(main())
