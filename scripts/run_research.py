import asyncio
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from rich.console import Console

from deep_research.state import ResearcherInputState, ResearcherOutputState
from deep_research.utils import display_markdown
from deep_research.workflows.research import build_researcher_graph


async def main():
    load_dotenv()

    config = RunnableConfig(
        run_id=uuid4(),
        configurable={"thread_id": str(uuid4())}
    )
    topic = "Research the impact of climate change on global agriculture and food security."
    input_state = ResearcherInputState(
        researcher_messages=[
            HumanMessage(content=topic)
        ],
        researcher_tool_calls=0,
        research_topic=topic,
    )

    graph = build_researcher_graph(
        checkpointer=InMemorySaver(),
        store=InMemoryStore()
    )
    result = cast(
        ResearcherOutputState,
        # pyrefly: ignore[no-matching-overload]
        await graph.ainvoke(input_state, config=config)
    )

    console = Console()
    display_markdown(result["compressed_research"], console=console)


if __name__ == "__main__":
    asyncio.run(main())
