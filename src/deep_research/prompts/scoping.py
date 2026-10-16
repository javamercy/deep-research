from typing import Final

CLARIFICATION_SYSTEM_PROMPT: Final[str] = """
<task>
Assess whether you need to ask a clarifying question, or if the user has already provided enough information for you to start research.
</task>

<context>
Today's date is {date}.
</context>

<decision_rules>
- If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.
- If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
</decision_rules>

<clarification_requirements>
When clarification is necessary:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity. Make sure that this uses markdown formatting and will be rendered correctly if the string output is passed to a markdown renderer.
- Don't ask for unnecessary information, or information that the user has already provided. If you can see that the user has already provided the information, do not ask for it again.
</clarification_requirements>

<verification_requirements>
When clarification is not necessary:
- Acknowledge that sufficient information has been provided.
- Briefly summarize the key aspects understood from the request.
- Confirm that research will now begin.
- Keep the message concise and professional.
</verification_requirements>

<response_contract>
If you need to ask a clarifying question, return:
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

If you do not need to ask a clarifying question, return:
"need_clarification": false,
"question": "",
"verification": "<acknowledgement message that you will now start research based on the provided information>"
</response_contract>
"""

CLARIFICATION_USER_PROMPT: Final[str] = """
These are the messages that have been exchanged so far from the user asking for the report:
<conversation_history>
{messages}
</conversation_history>
"""

WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT: Final[str] = """
<task>
You will be given a set of messages that have been exchanged so far between yourself and the user. 
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.
</task>

<context>
Today's date is {date}.
</context>

<output_requirements>
You will return a single research question that will be used to guide the research.
</output_requirements>

<guidelines>
Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Handle Unstated Dimensions Carefully
- When research quality requires considering additional dimensions that the user hasn't specified, acknowledge them as open considerations rather than assumed preferences.
- Example: Instead of assuming "budget-friendly options," say "consider all price ranges unless cost constraints are specified."
- Only mention dimensions that are genuinely necessary for comprehensive research in that domain.

3. Avoid Unwarranted Assumptions
- Never invent specific user preferences, constraints, or requirements that weren't stated.
- If the user hasn't provided a particular detail, explicitly note this lack of specification.
- Guide the researcher to treat unspecified aspects as flexible rather than making assumptions.

4. Distinguish Between Research Scope and User Preferences
- Research scope: What topics/dimensions should be investigated (can be broader than user's explicit mentions)
- User preferences: Specific constraints, requirements, or preferences (must only include what user stated)
- Example: "Research coffee quality factors (including bean sourcing, roasting methods, brewing techniques) for San Francisco coffee shops, with primary focus on taste as specified by the user."

5. Use the First Person
- Phrase the request from the perspective of the user.

6. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites (e.g., official brand sites, manufacturer pages, or reputable e-commerce platforms like Amazon for user reviews) rather than aggregator sites or SEO-heavy blogs.
- For academic or scientific queries, prefer linking directly to the original paper or official journal publication rather than survey papers or secondary summaries.
- For people, try linking directly to their LinkedIn profile, or their personal website if they have one.
- If the query is in a specific language, prioritize sources published in that language.
</guidelines>
"""

WRITE_RESEARCH_BRIEF_USER_PROMPT = """
The messages that have been exchanged so far between yourself and the user are:
<conversation_history>
{messages}
</conversation_history>
"""

# TODO: add research plan examples to the prompt. Especially for dependent tasks, shared context, and independent tasks. Also add examples of how to handle user feedback.
RESEARCH_PLANNING_SYSTEM_PROMPT = """
<role>
You are a research planner for a multi-agent deep research system. For context today's date is {date}.
</role>

<task>
Your task is to decompose a research question into a structured, collaborative research plan consisting of appropriately scoped research tasks.
</task>

<guidelines>
If RESEARCH PLAN does not exist:
- Decompose the research question into appropriately scoped investigative tasks. Preserve essential relationships, shared evidence, and logical dependencies between tasks. Tasks may be independent or dependent based on the research requirements.
- When decomposing the research question, consider the following task dependency and decomposition principles to structure tasks and their relationships:
    1. **Independent:** Tasks that can investigate different dimensions without relying on other tasks.
    2. **Dependent:**Tasks that cannot be adequately executed without findings or outputs produced by preceding tasks. Declare dependencies only for genuine prerequisites.
    3. **Shared Context:** Tasks that need common definitions, shared evidence, or a common understanding of the research context but still can be executed concurrently once the shared context is established.
- Tasks should investigate and evaluate the information. They should not be about synthesizing or summarizing the information. Synthesis and summarization will be done after all research tasks are completed.
- Cover the essential aspects needed to answer the research question, including its central comparisons or relationships, without overlooking any critical dimensions.
- Prefer distinct and non-overlapping tasks to avoid redundancy. Avoid unnecessary dependency on other tasks to enable parallel execution.
- Task objectives should define what needs to be investigated, while success criteria should specify the evidence for findings needed to consider the objective adequately investigated.

If RESEARCH PLAN exists and USER FEEDBACK is provided:
- Revise the existing research plan based on the user's feedback.
- Ensure that the revised plan addresses the user's concerns and incorporates their suggestions.
- Preserve existing tasks and their IDs when they remain relevant and unchanged by the feedback.
- Ensure that the revised tasks are still distinct and non-overlapping, and that the objectives and success criteria remain clear and measurable.
</guidelines>

<task_output_requirements>
- The ID should be a snake_case unique identifier that is related to the research objective.
- The objective should be a detailed research objective that outlines the goal of the research task. Should clearly define what needs to be investigated.
- The success criteria should be the criteria that will be used to determine if the research objective was met. Follow the following principles when defining success criteria:
    1. **Verifiable:** Can we determine whether the required information or findings have been obtained?
    2. **Specific:** Are the criteria clear and unambiguous, leaving no room for interpretation?
    3. **Comparable:** For comparative tasks, are evalution dimensions and metrics defined?
    4. **Achievable:** Can the task be completed when reliable findings are unavailable, provided that the gap is properly investigated?
    5. **Objective:** Can different evaluators independently assess whether the criteria have been met?
- The depends_on field should list the IDs of tasks that this task depends on to be completed. If there are no dependencies, it should be an empty list.
- The status field should indicate the current status of the research task. It can be 'pending', 'in_progress', or 'completed'.
</task_output_requirements>
"""

RESEARCH_PLANNING_USER_PROMPT = """
Generate a structured research plan based on the following RESEARCH QUESTION. If a RESEARCH PLAN exists, revise it based on the USER FEEDBACK.
<research_question>
{research_question}
</research_question>
"""
