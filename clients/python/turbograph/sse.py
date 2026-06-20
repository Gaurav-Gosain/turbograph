"""A small server-sent events (SSE) parser.

turbograph streams its chat, entity-build, community-build, and model-pull
endpoints as SSE. Each event is a block of lines terminated by a blank line::

    event: token
    data: {"text": "hello"}

    event: done
    data: {"done": true}

The wire format groups consecutive "data:" lines into one payload joined by
newlines (per the SSE specification). turbograph emits exactly one data line per
event, but this parser handles the general case so it stays correct.

The parser is deliberately decoupled from the transport so it can be unit-tested
against a plain iterator of byte chunks without a live server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional


@dataclass
class SSEEvent:
    """A single parsed server-sent event.

    Attributes:
        event: the event name (the "event:" field), defaulting to "message"
            when the stream omits it, matching the SSE specification.
        data: the parsed JSON payload of the "data:" field. turbograph always
            sends JSON objects, so this is a dict in practice; if a data line is
            not valid JSON it is returned as the raw string instead.
        raw: the undecoded data string, kept for debugging.
        id: the event id (the "id:" field), if any.
    """

    event: str = "message"
    data: Any = None
    raw: str = ""
    id: Optional[str] = None


def parse_sse(chunks: Iterable[bytes]) -> Iterator[SSEEvent]:
    """Parse an iterable of byte chunks into a stream of SSEEvent objects.

    The chunks need not align to event or line boundaries; this buffers and
    splits on blank lines, so it is safe to feed it whatever the socket yields.

    Args:
        chunks: an iterable of bytes (for example, reads from an HTTP response).

    Yields:
        SSEEvent for every complete event in the stream.
    """
    buffer = ""
    for chunk in chunks:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="replace")
        else:
            buffer += chunk
        # Normalize line endings so CRLF streams split the same as LF streams.
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event = _parse_block(block)
            if event is not None:
                yield event
    # Flush a trailing event that arrived without a terminating blank line.
    tail = buffer.strip()
    if tail:
        event = _parse_block(tail)
        if event is not None:
            yield event


def _parse_block(block: str) -> Optional[SSEEvent]:
    """Parse one event block (the text between blank-line separators)."""
    name = "message"
    event_id: Optional[str] = None
    data_lines = []
    for line in block.split("\n"):
        if line == "" or line.startswith(":"):
            # Blank lines inside a block and comment lines are ignored.
            continue
        field_name, _, value = line.partition(":")
        # A single leading space after the colon is part of the format, not data.
        if value.startswith(" "):
            value = value[1:]
        if field_name == "event":
            name = value
        elif field_name == "data":
            data_lines.append(value)
        elif field_name == "id":
            event_id = value
    if not data_lines and name == "message":
        return None
    raw = "\n".join(data_lines)
    try:
        data: Any = json.loads(raw) if raw else None
    except (ValueError, TypeError):
        data = raw
    return SSEEvent(event=name, data=data, raw=raw, id=event_id)
