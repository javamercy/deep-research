---
name: python-docstrings
description: Write and maintain accurate, proportional PEP 257 / Google-style docstrings whenever Python code in this project is added, changed, refactored, or reviewed. Covers ordinary code and model-facing docstrings (LLM tool descriptions, structured-output schemas) where the docstring is effectively part of a prompt. Use for any Python edit, docstring review, or "document this" request.
---

# Python docstrings

A docstring is a contract: what callers can rely on and what maintainers must preserve. In this project some docstrings are also **sent to an LLM**, so for those, wording changes behavior. Optimize for accuracy first, usefulness second, brevity third.

## Orient before writing

The project's layout and architecture change over time, so do not rely on a remembered file map. Before documenting anything, read the code you are changing, its existing docstrings, and enough of its callers, types, and graph wiring to confirm every behavior you plan to describe.

## Apply on every relevant change

1. Add or revise docstrings for new and changed interfaces, non-obvious behavior, and contracts.
2. Delete or fix claims that are no longer true. A stale docstring is worse than none.
3. Update affected docstrings when behavior changes even if the signature does not, including changes driven by configuration, defaults, or refactoring.
4. Keep the change scoped. Do not rewrite unrelated docstrings or comments, and do not add docstrings to untouched code just for coverage.

A change does **not** require a new docstring on every touched function. It requires checking that the affected documentation is accurate and sufficient.

## Style

Follow PEP 257 formatting and Google style sections.

- Use `"""triple double quotes"""`. Put the summary on the first line, keep it to one line, and end it with a period.
- Function summaries use the imperative mood (`Return ...`, `Build ...`, `Validate ...`). If a file already consistently uses another mood, match it.
- Do not open with filler such as `This function ...`, `Tool for ...`, `Schema for ...`, or `Helper to ...`. State what the thing does or represents.
- Separate the summary from the body with a blank line. Indent the body at the same level as the opening quotes.
- Never repeat type annotations, parameter names, or line-by-line implementation steps. Explain meaning, units, valid ranges, invariants, ordering, defaults, and what callers can rely on.

### Choosing the level of detail

| Situation | Docstring |
| --- | --- |
| Behavior is obvious from the name and signature | One line, or none for trivial private helpers |
| Ordering, defaults, edge cases, side effects, I/O, caching, concurrency, or a non-obvious design decision | Summary plus a short descriptive body |
| Callers need per-parameter semantics, a non-obvious return shape, or must handle errors | Add the matching Google sections |

### Google sections

Use these section names: `Args:`, `Returns:`, `Yields:`, `Raises:`, `Attributes:`. Never add an empty or purely boilerplate section.

- `Args:` uses `name: description`, with continuation lines indented. Document only the parameters that need explanation, except when a tool parser needs all of them (see below). Describe `*args` and `**kwargs` by their role.
- `Returns:` describes the meaning and shape of the value. Omit it for `None`, and when the summary already says what is returned.
- `Yields:` replaces `Returns:` for generators and async generators.
- `Raises:` lists only exceptions the code raises on purpose or deliberately lets propagate as part of its contract, not every exception a dependency might raise.
- `Attributes:` goes in class docstrings, for public attributes that need explanation. Document `__init__` arguments in either the class docstring or `__init__`, not both.
- A `@property` is documented like an attribute (`"""The active session ID."""`), not `Returns the ...`.
- Methods decorated with `@override` need no docstring unless they change the parent's contract.

### Comments vs docstrings

Docstrings describe the contract of a module, class, or callable. Inline comments explain local implementation choices (why this lock, why this order). Keep comments that already exist unless your change makes them wrong.

## Model-facing docstrings (critical)

When a docstring or field description is turned into a JSON schema for an LLM, write it as **prompt text for the model**, not as maintainer notes. Before relying on the details below, check them against the installed library source, which is the ground truth for this project.

**Function tools (LangChain `@tool`)**

- Without `parse_docstring=True`, the entire docstring becomes the tool description, including any `Args:` or `Returns:` text.
- With `parse_docstring=True`, the text before `Args:` becomes the description, and each `Args:` entry becomes that parameter's schema description. `Returns:` is not used as the description. An invalid Google docstring (for example, no `Args:` block when the tool has model-visible parameters) raises at import time by default.
- Injected parameters (runtime, store, config, and arguments marked as injected) are hidden from the model. You may document them for maintainers, but keep the description from depending on them.

**Pydantic or class-based tools and structured-output schemas**

- The class docstring becomes the schema `description`, and `Field(description=...)` becomes each property's description. Both reach the model.
- Do not put `Attributes:` sections, maintainer notes, or implementation details in these class docstrings. Put field semantics in `Field(description=...)`, and keep maintainer-only rationale in comments.
- Nested models carry their own docstrings into the schema, so they need the same care.

**Writing guidance for model-facing text**

- Say what the tool or schema is for and when to use it, plus the constraints that matter (scope, required level of detail, allowed values, units, limits).
- Keep wording consistent with the prompts that refer to the tool or schema by name.
- Treat edits to this text as prompt changes. Mention them in your final message, because they can affect agent behavior and evaluation results.

## Guidance by kind of code

Identify which roles the changed code plays, whatever its current file or directory. Document what applies:

| Role | Document when relevant |
| --- | --- |
| Graph builders and nodes | Purpose, the state fields read, the partial update or `Command` returned, routing and termination conditions, interrupts and what resumes them, and reducer-sensitive fields. Explain non-obvious transitions instead of restating every edge. Note required runtime dependencies such as a store or checkpointer. |
| State schemas | Purpose, non-obvious fields, invariants, and merge or reducer semantics. These are usually not model-facing, so `Attributes:` or comments are fine. |
| LLM tools and structured-output schemas | See **Model-facing docstrings**. |
| Prompt templates | The purpose of the template, the variables it requires, and the output format it expects, especially where the template and its caller must agree. Do not paraphrase the prompt body. |
| Model factories and configuration | Defaults, override precedence (for example runtime config, then environment, then defaults), and provider-specific assumptions. |
| External I/O (search, HTTP, model calls) | What is called, limits, retries, normalization, deduplication, error handling, and cost or rate-limit side effects. |
| Storage and memory | Identity and keys, namespaces, overwrite or merge semantics, ordering, and concurrency guarantees. |
| Formatting and rendering helpers | Output format, escaping, ordering, omissions, and trust boundaries (untrusted web content inside prompts). |
| Entrypoints, scripts, and import-time modules | Purpose, required inputs or environment, material side effects (paid API calls, writes), and anything executed at import. |
| Evaluations | What the dataset, target, or evaluator measures, the scoring scale and how to interpret it, and reproducibility assumptions. |

### Depth by visibility

- **Public interfaces, graph nodes, and tools:** make sure a useful docstring exists, and add detail when callers or maintainers need it.
- **Private helpers:** one line if it adds context; more only for real complexity; nothing for trivial restatements.
- **Modules:** a concise module docstring when the module's purpose is not obvious from its name.
- **Tests and evaluations:** explain unusual fixtures, scoring logic, or regression context. Avoid boilerplate like `"""Test X."""`.

## Examples

These names are illustrative. They show the **level of documentation**, not contracts that exist in the project.

A simple helper needs no sections:

```python
def normalize_query(query: str) -> str:
    """Collapse whitespace and casefold a query for matching."""
    return " ".join(query.split()).casefold()
```

A graph node documents its state contract:

```python
async def gather_evidence(state: ResearchState) -> dict[str, object]:
    """Collect sources for the active research question.

    Reads the question and previously seen source IDs from state and skips
    sources already collected. `sources` uses an additive reducer, so only
    new items are returned.

    Returns:
        A partial state update with the newly collected `sources`.
    """
```

A reusable function documents its contract:

```python
def select_sources(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Select distinct sources in relevance order.

    Deduplicates by normalized URL, keeping the first (highest-ranked)
    occurrence.

    Args:
        results: Candidates ordered from most to least relevant.
        limit: Maximum number of distinct sources to keep.

    Returns:
        At most `limit` results, in their original relative order.

    Raises:
        ValueError: If `limit` is negative.
    """
```

A model-facing tool with `parse_docstring=True`, where every line is prompt text:

```python
@tool(parse_docstring=True)
async def web_search(query: str, runtime: ToolRuntime) -> str:
    """Search the web and return summarized sources with citation IDs.

    Use one focused query per call. Prefer specific entities, dates, and
    terms over broad questions.

    Args:
        query: A specific search query, not a full research question.
        runtime: Injected execution context; not visible to the model.
    """
```

A structured-output schema keeps maintainer notes out of the schema:

```python
class Finding(BaseModel):
    """A single factual claim supported by one or more sources."""

    claim: str = Field(description="One verifiable statement, without hedging.")
    source_ids: list[str] = Field(description="IDs of the sources that support the claim.")
```

## Completion check

Before finishing any Python change, confirm:

- New or changed public interfaces, graph nodes, tools, and schemas are documented appropriately.
- Docstrings match the implemented behavior: state updates, routing, tool inputs, side effects, and raised errors.
- Model-facing text is written for the model, contains no maintainer notes, and stays consistent with the prompts. Any change to it is called out to the user.
- No redundant sections, repeated type hints, or filler openings were introduced.
- No stale docstrings remain in the affected code, and unrelated code was left alone.
- The project's lint checks pass for the changed code. Fix docstring findings you introduced, and report pre-existing violations instead of expanding the change to fix them.
