from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console

from deep_research.state import AgentInputState
from deep_research.utils import display_markdown, display_messages
from deep_research.workflows.scoping import build_scoping_graph


async def main():
    load_dotenv()
    config = RunnableConfig(configurable={"thread_id": str(uuid4())})
    input_state = AgentInputState(
        messages=[
            HumanMessage(
                content="I want to research kebab restaurants in Istanbul."
            )
        ]
    )
    graph = build_scoping_graph(checkpointer=InMemorySaver())
    result = await graph.ainvoke(input=input_state, config=config)

    console = Console()

    display_messages(result["messages"], console=console)

    input_state = AgentInputState(
        messages=[
            HumanMessage(
                content="I am into the most delicious and authentic ones. The area is not important. Any type of kebab is fine."
            )
        ]
    )

    result = await graph.ainvoke(input=input_state, config=config)

    while interrupts := result.get("__interrupt__"):
        request = interrupts[0].value
        options = request["options"]

        feedback = ""
        while True:
            action = (
                await asyncio.to_thread(
                    console.input, f"Choose an action {options}: "
                )
            ).strip().lower()

            if action not in options:
                console.print("Invalid decision.")
                continue

            if action == "revise":
                feedback = await asyncio.to_thread(
                    console.input, "Revision feedback: "
                ).strip()

            if not feedback:
                console.print("Please provide feedback for revision.")
                continue

            break

        result = await graph.ainvoke(
            Command(
                resume={
                    "action": action,
                    "feedback": feedback
                }
            ),
            config=config
        )

    display_messages(result["messages"], console=console)
    display_markdown(result["research_brief"], console=console)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
