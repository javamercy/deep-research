import asyncio
from pathlib import Path
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from deep_research.state import SupervisorState
from deep_research.utils import display_messages
from deep_research.workflows.supervisor import build_supervisor_graph


async def main():
    load_dotenv()
    brief_path = Path(__file__).parent / "examples" / "research_brief.txt"
    research_brief = brief_path.read_text(encoding="utf-8")

    config = RunnableConfig(
        configurable={"thread_id": uuid4()},
        recursion_limit=50
    )
    graph = build_supervisor_graph(checkpointer=InMemorySaver())

    messages = [HumanMessage(content=research_brief)]

    result = cast(
        SupervisorState,
        await graph.ainvoke({"supervisor_messages": messages}, config=config)
    )

    console = Console()
    display_messages(result["supervisor_messages"], console=console)


if __name__ == "__main__":
    asyncio.run(main())
