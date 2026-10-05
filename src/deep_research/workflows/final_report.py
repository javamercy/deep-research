from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration
from deep_research.models import init_openrouter_model
from deep_research.prompts.final_report import GENERATE_FINAL_REPORT_SYSTEM_PROMPT, GENERATE_FINAL_REPORT_USER_PROMPT
from deep_research.state import AgentState
from deep_research.utils import get_today_str


async def generate_final_report(
        state: AgentState,
        config: RunnableConfig
) -> dict:
    """Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    configuration = Configuration.from_runnable_config(config)
    writer_model = init_openrouter_model(llm_config=configuration.writer_llm_config)

    notes = state["notes"]
    findings = "\n".join(notes)

    messages = [
        SystemMessage(
            content=GENERATE_FINAL_REPORT_SYSTEM_PROMPT
            .format(date=get_today_str())
        ),
        HumanMessage(
            content=GENERATE_FINAL_REPORT_USER_PROMPT
            .format(findings=findings, research_brief=state["research_brief"])
        )
    ]
    response = await writer_model.ainvoke(messages, config=config)

    return {
        "final_report": response.text,
        "messages": [AIMessage(f"Here is the final report:\n\n{response.text}")],
    }
