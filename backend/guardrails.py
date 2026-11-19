"""Minimal input/output guardrails for a local research preview.

These checks are not a full safety system and do not replace red-team testing.
"""
from __future__ import annotations
import re

MAX_PROMPT_CHARS=350
BLOCKED_PATTERNS=[re.compile(r"(?:-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)",re.I)]

def validate_prompt(text:str)->str:
    value=text.strip()
    if not value: raise ValueError('Message cannot be empty.')
    if len(value)>MAX_PROMPT_CHARS: raise ValueError(f'Message exceeds {MAX_PROMPT_CHARS} characters.')
    if any(p.search(value) for p in BLOCKED_PATTERNS): raise ValueError('Do not submit private keys or credentials.')
    return value

def add_research_notice(answer:str)->str:
    # Avoid rewriting generated content; caller displays a persistent disclaimer in UI.
    return answer.strip()
