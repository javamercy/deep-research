from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from deep_research.configuration import Configuration
from deep_research.state import AgentInputState, AgentState
from deep_research.utils import display_markdown, display_messages
from deep_research.workflows.scoping import build_scoping_graph


async def main():
    config = RunnableConfig(configurable={"thread_id": str(uuid4())})
    context = Configuration()
    input_state = AgentInputState(
        messages=[
            HumanMessage(
                content="I want to research kebab restaurants in Istanbul."
            )
        ]
    )
    graph = build_scoping_graph(checkpointer=InMemorySaver())
    # TODO: see workflows/scoping.py:99
    # pyrefly: ignore [no-matching-overload]
    result: AgentState = await graph.ainvoke(
        input=input_state,
        context=context,
        config=config,
    )

    console = Console(width=150)

    display_messages(result["messages"], console=console)

    input_state = AgentInputState(
        messages=[
            HumanMessage(
                content="I am into the most delicious and authentic ones. The area is not important. Any type of kebab is fine."
            )
        ]
    )
    # pyrefly: ignore [no-matching-overload]
    result: AgentState = await graph.ainvoke(
        input=input_state,
        context=context,
        config=config,
    )

    display_messages(result["messages"], console=console)
    display_markdown(result["research_brief"], console=console)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
