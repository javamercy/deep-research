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

RESEARCH_PLANNING_SYSTEM_PROMPT = """
<role>
You are a research planner for a multi-agent deep research system. For context today's date is {date}.
</role>

<task>
Your task is to transform a research question into a structured,  collaborative research plan consisting of appropriately scoped research tasks.
</task>

<guidelines>
If RESEARCH PLAN does not exist:
- Decompose the research question into smaller, manageable research tasks that can be executed **mostly** independently.
- Tasks should investigate and evaluate the information. They should not be about synthesizing or summarizing the information. Synthesis and summarization will be done after all research tasks are completed.
- Cover the essential aspects needed to answer the research question, including its central comparisons or relationships, without overlooking any critical dimensions.
- Prefer distinct and non-overlapping tasks to avoid redundancy.
- Task objectives should define what needs to be investigated, while success criteria should specify the evidence for findings needed to consider the objective adequately investigated.

If RESEARCH PLAN exists and USER FEEDBACK is provided:
- Revise the existing research plan based on the user's feedback.
- Ensure that the revised plan addresses the user's concerns and incorporates their suggestions.
- Preserve existing tasks and their IDs when they remain relevant and unchanged by the feedback.
- Ensure that the revised tasks are still distinct and non-overlapping, and that the objectives and success criteria remain clear and measurable.
</guidelines>


<examples>
  <example id="1" type="initial_planning">
    <research_question>
      Which kebab restaurants in Istanbul offer the most
      authentic and high-quality experience, prioritizing
      food quality over tourist popularity?
    </research_question>

    <research_tasks>
      <task>
        <id>kebab_traditions</id>
        <objective>
          Identify the characteristics of authentic kebab
          styles served in Istanbul.
        </objective>
        <success_criteria>
          Establish distinguishing regional styles and
          preparation methods using credible Turkish
          culinary sources.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>local_recommendations</id>
        <objective>
          Identify kebab restaurants recommended by
          knowledgeable local food writers and critics.
        </objective>
        <success_criteria>
          Find credible recommendations supported by
          independent editorial coverage and evidence
          of culinary specialization.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>quality_assessment</id>
        <objective>
          Evaluate the food quality and consistency
          of recommended restaurants.
        </objective>
        <success_criteria>
          Compare substantive recent dining experiences,
          identifying consistent strengths, weaknesses,
          and conflicting assessments.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>practical_verification</id>
        <objective>
          Verify the current dining details of the
          strongest restaurant candidates.
        </objective>
        <success_criteria>
          Confirm locations, operating status, and
          relevant menu offerings where reliable
          information is available.
        </success_criteria>
        <status>pending</status>
      </task>
    </research_tasks>
  </example>

  <example id="2" type="collaborative_plan_revision">
    <research_question>
      In 2026, should enterprises use retrieval-augmented
      generation (RAG) or long-context language models for
      internal knowledge-base question answering?
    </research_question>

    <previous_plan>
      <task>
        <id>retrieval_accuracy</id>
        <objective>
          Compare RAG and long-context approaches on
          factual accuracy and retrieval performance.
        </objective>
        <success_criteria>
          Identify relevant empirical evaluations, their
          methodologies, results, and limitations.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>cost_latency</id>
        <objective>
          Compare inference cost, latency, and scalability
          under realistic enterprise workloads.
        </objective>
        <success_criteria>
          Establish comparable cost and latency trade-offs
          with explicit workload assumptions.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>operational_constraints</id>
        <objective>
          Evaluate document freshness, access control,
          and context-window limitations.
        </objective>
        <success_criteria>
          Identify operational constraints and failure
          modes that materially influence architecture
          selection.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>production_evidence</id>
        <objective>
          Investigate real-world deployments of RAG,
          long-context, and hybrid architectures.
        </objective>
        <success_criteria>
          Find documented production experiences and
          distinguish measured outcomes from unsupported
          vendor claims.
        </success_criteria>
        <status>pending</status>
      </task>
    </previous_plan>

    <user_feedback>
      Our knowledge base contains over 100,000 documents,
      including sensitive customer information. Documents
      change several times a day, and users must only
      access documents they are authorized to see.
      Prioritize those requirements and include a
      comparison with hybrid approaches.
    </user_feedback>

    <revised_plan>
      <task>
        <id>retrieval_accuracy</id>
        <objective>
          Compare RAG, long-context, and hybrid approaches
          on factual accuracy and retrieval performance.
        </objective>
        <success_criteria>
          Identify relevant empirical evaluations,
          including evidence about performance on large
          document collections.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>cost_latency</id>
        <objective>
          Compare cost, latency, and scalability for
          knowledge bases exceeding 100,000 documents.
        </objective>
        <success_criteria>
          Establish comparable trade-offs using realistic
          document volumes, query loads, and context sizes.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>access_control</id>
        <objective>
          Investigate document-level authorization and
          sensitive-data exposure risks across architectures.
        </objective>
        <success_criteria>
          Identify viable permission-enforcement patterns,
          security limitations, and documented failure risks.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>document_freshness</id>
        <objective>
          Compare how architectures handle frequently
          updated enterprise documents.
        </objective>
        <success_criteria>
          Evaluate ingestion, update propagation, indexing
          requirements, and risks of outdated answers.
        </success_criteria>
        <status>pending</status>
      </task>

      <task>
        <id>production_evidence</id>
        <objective>
          Investigate enterprise deployments of RAG,
          long-context, and hybrid architectures.
        </objective>
        <success_criteria>
          Find documented production experiences relevant
          to large, frequently updated, access-controlled
          knowledge bases.
        </success_criteria>
        <status>pending</status>
      </task>
    </revised_plan>
  </example>
</examples>


<output_requirements>
- Every research task should have a unique identifier that is related to the research objective.
<output_requirements>
"""

RESEARCH_PLANNING_USER_PROMPT = """
Generate a structured research plan based on the following RESEARCH QUESTION. If a RESEARCH PLAN exists, revise it based on the USER FEEDBACK.
<research_question>
{research_question}
</research_question>
"""
