"""Shared utility helpers."""

from datetime import datetime


def get_today_str():
    """Return the current local date formatted for prompts (e.g., `Mon Nov 23, 2026`)."""
    return datetime.now().strftime("%a %b %-d, %Y")
