import json
from collections.abc import Sequence
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

_MESSAGE_STYLES = {
    "human": ("🧑", "Human", "blue"),
    "ai": ("🤖", "Assistant", "green"),
    "tool": ("🔧", "Tool Output", "yellow"),
    "system": ("⚙️", "System", "cyan"),
}


def get_today_str():
    """Get current date in a human-readable format."""

    return datetime.now().strftime("%a %b %-d, %Y")


def display_markdown(content: str, *, console: Console | None = None) -> None:
    """Render plain text or Markdown content in the terminal."""

    output = console or Console(width=120)
    output.print(Markdown(content, code_theme="ansi_dark"))


def _json(value: object) -> str:
    """Serialize values for readable terminal output."""

    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _tool_call_key(name: str, arguments: object, call_id: object) -> tuple[str, str]:
    """Build a stable key used to avoid rendering the same tool call twice."""

    if call_id:
        return "id", str(call_id)
    return name, _json(arguments)


def _tool_call_panel(
    name: str,
    arguments: object,
    call_id: object = None,
    error: object = None,
) -> Panel:
    """Create a compact, structured rendering of one tool call."""

    details: list[RenderableType] = [
        Syntax(
            _json(arguments),
            "json",
            theme="ansi_dark",
            word_wrap=True,
            background_color="default",
            padding=(0, 1),
        )
    ]
    if call_id:
        details.append(Text(f"Call ID: {call_id}", style="dim"))
    if error:
        details.append(Text(f"Error: {error}", style="bold red"))

    return Panel(
        Group(*details),
        title=Text.assemble(("🔧 ", "magenta"), (name, "bold magenta")),
        title_align="left",
        border_style="magenta",
        padding=(0, 1),
    )


def _message_body(message: BaseMessage) -> Group:
    """Turn a LangChain message body into Rich renderables."""

    renderables: list[RenderableType] = []
    rendered_tool_calls: set[tuple[str, str]] = set()
    content = message.content

    if isinstance(content, str):
        if content:
            renderables.append(Text(content))
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                renderables.append(Text(block))
                continue

            block_type = block.get("type")
            if block_type in {"text", "plain_text"}:
                renderables.append(Text(str(block.get("text", ""))))
            elif block_type in {"tool_use", "tool_call"}:
                name = str(block.get("name", "Unknown tool"))
                arguments = block.get("input", block.get("args", {}))
                call_id = block.get("id")
                rendered_tool_calls.add(_tool_call_key(name, arguments, call_id))
                renderables.append(_tool_call_panel(name, arguments, call_id))
            else:
                label = str(block_type or "content")
                renderables.append(
                    Panel(
                        Syntax(
                            _json(block),
                            "json",
                            theme="ansi_dark",
                            word_wrap=True,
                            background_color="default",
                        ),
                        title=f"{label.replace('_', ' ').title()} block",
                        title_align="left",
                        border_style="dim",
                    )
                )

    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            name = tool_call.get("name", "Unknown tool")
            arguments = tool_call.get("args", {})
            call_id = tool_call.get("id")
            key = _tool_call_key(name, arguments, call_id)
            if key not in rendered_tool_calls:
                renderables.append(_tool_call_panel(name, arguments, call_id))
                rendered_tool_calls.add(key)

        for invalid_call in message.invalid_tool_calls:
            name = invalid_call.get("name") or "Invalid tool call"
            arguments = invalid_call.get("args", {})
            call_id = invalid_call.get("id")
            key = _tool_call_key(name, arguments, call_id)
            if key not in rendered_tool_calls:
                renderables.append(
                    _tool_call_panel(
                        name,
                        arguments,
                        call_id,
                        invalid_call.get("error") or "Invalid arguments",
                    )
                )
                rendered_tool_calls.add(key)

    if not renderables:
        renderables.append(Text("No content", style="dim italic"))

    return Group(*renderables)


def _message_subtitle(message: BaseMessage) -> Text | None:
    """Return useful message metadata without cluttering the body."""

    metadata: list[str] = []
    if message.name:
        metadata.append(message.name)
    if isinstance(message, ToolMessage):
        metadata.append(f"call {message.tool_call_id}")
        if message.status == "error":
            metadata.append("error")

    if not metadata:
        return None
    return Text(" · ".join(metadata), style="dim")


def display_messages(
    messages: Sequence[BaseMessage],
    *,
    console: Console | None = None,
) -> None:
    """Display LangChain messages as a readable, role-aware conversation."""

    output = console or Console(width=120)
    count = len(messages)
    noun = "message" if count == 1 else "messages"
    output.print(Rule(f"Conversation · {count} {noun}", style="bright_black"))

    for index, message in enumerate(messages):
        icon, label, color = _MESSAGE_STYLES.get(
            message.type,
            ("📝", message.type.replace("_", " ").title(), "white"),
        )
        output.print(
            Panel(
                _message_body(message),
                title=Text(f"{icon} {label}", style=f"bold {color}"),
                title_align="left",
                subtitle=_message_subtitle(message),
                subtitle_align="right",
                border_style=color,
                padding=(1, 2),
            )
        )
        if index < count - 1:
            output.print()
