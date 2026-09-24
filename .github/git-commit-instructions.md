# Commit Message Instructions

Generate professional commit messages following the Conventional Commits specification.

Format:
<type>(<scope>): <description>

Rules:

- Use an appropriate type:
    - feat: new functionality
    - fix: bug fixes
    - refactor: internal code restructuring without behavior changes
    - style: formatting, import ordering, and lint-only changes
    - perf: performance improvements
    - docs: documentation changes
    - test: adding or updating tests
    - build: build system or dependency changes
    - ci: CI/CD configuration changes
    - chore: routine maintenance
- Choose the commit type based on the actual changes, not the files modified.
- Include a scope when the affected component or module is identifiable.
- Use imperative mood and lowercase descriptions.
- Keep the subject line within 72 characters, without a trailing period.
- Describe what changed and why, not merely which files changed.
- For complex changes, add a blank line followed by a concise body explaining the motivation and key changes.
- Use `!` for breaking changes and include a `BREAKING CHANGE:` footer when applicable.
- Reference issue IDs only when explicitly provided.
- Base the message strictly on the actual changes; never invent details.
- Output only the commit message, without Markdown formatting or explanations.