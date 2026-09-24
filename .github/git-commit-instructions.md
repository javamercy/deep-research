# Commit Message Instructions

Generate professional commit messages following the Conventional Commits specification.

Format:
<type>(<scope>): <description>

Rules:

- Use an appropriate type: feat, fix, refactor, perf, docs, test, build, ci, or chore.
- Include a scope when the affected component or module is identifiable.
- Use imperative mood and lowercase descriptions.
- Keep the subject line within 72 characters, without a trailing period.
- Describe what changed and why, not merely which files changed.
- For complex changes, add a blank line followed by a concise body explaining the motivation and key changes.
- Use `!` for breaking changes and include a `BREAKING CHANGE:` footer when applicable.
- Reference issue IDs only when explicitly provided.
- Base the message strictly on the actual changes; never invent details.
- Output only the commit message, without Markdown formatting or explanations.