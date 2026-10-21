# Engineering Learning Notebook (`NOTES.md`)

## Purpose & Instructions for Agents

### Aim of This File
This document is a **persistent engineering notebook** maintained across sessions. In this repository, the developer works as a learner exploring advanced agentic AI systems, and the AI assistant acts as an expert mentor/teacher. Whenever a design discussion, code review, or debugging session reveals a **non-obvious architectural principle, production failure mode, frontier research pattern, or framework internal gotcha**, the agent should record it here.

### Developer Baseline (What NOT to Add)
Do **not** record foundational or routine topics that the developer already practices in this repository, such as:
- Standard Python 3.13, `asyncio`, type annotations, or basic Pydantic v2 model creation.
- Standard LangGraph wiring (`StateGraph`, `Command(goto=..., update=...)`, `ToolNode`, `interrupt`, reducers like `operator.add` and `add_messages`).
- Basic LLM API usage, tool binding, or basic `str.format` prompt templating.

### Rules for Agents Adding New Notes
1. **Self-Contained Context**: Write every entry so that a reader or agent in a fresh conversation (with no access to past chat transcripts) can immediately understand the scenario, the problem, and the rationale.
2. **Focus on "Why", Not Just "What"**: Explain the underlying failure mode, trade-off, or system dynamics—not just a code snippet.
3. **Categorize Cleanly**: Add new insights under an existing thematic section below (or create a new top-level numbered section if introducing a distinct domain).
4. **Entry Template**: Use a consistent structure for each note:
   - **Context / Problem**: What we were trying to achieve and what naive approach fails (or is suboptimal).
   - **Principle / Solution**: What experts and production systems do instead.
   - **Reasoning / Trade-offs**: Why the solution works and when to use alternatives.

---
