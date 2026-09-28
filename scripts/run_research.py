import asyncio
from pathlib import Path
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from deep_research.state import ResearcherOutputState, ResearcherState
from deep_research.utils import display_messages
from deep_research.workflows.research import build_research_graph


async def main():
    load_dotenv()

    brief_path = Path(__file__).parent / "examples" / "research_brief.txt"
    research_brief = brief_path.read_text(encoding="utf-8")

    config = RunnableConfig(configurable={"thread_id": str(uuid4())})

    researcher_state = ResearcherState(
        researcher_messages=[
            HumanMessage(content=research_brief)
        ],
        tool_call_iterations=0,
        research_topic="Researching the most delicious and authentic kebab restaurants in Istanbul, Turkey.",
        compressed_research="",
        raw_notes=[]
    )

    graph = build_research_graph(checkpointer=InMemorySaver())
    result = cast(
        ResearcherOutputState,
        await graph.ainvoke(researcher_state, config=config)
    )

    console = Console()
    display_messages(result["researcher_messages"], console=console)


if __name__ == "__main__":
    asyncio.run(main())
