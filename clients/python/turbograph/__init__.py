"""turbograph - a pure standard library Python client for the turbograph graph-RAG server.

turbograph is a fast, dependency-free graph-RAG server. This client mirrors that
ethos: it uses only the Python standard library (urllib, json, base64, dataclasses,
typing) and works on Python 3.9+.

Typical use::

    from turbograph import Client

    tg = Client("http://localhost:8080", bucket="default")
    tg.ingest_text("doc1", "The quick brown fox jumps over the lazy dog.")
    for hit in tg.query("what does the fox do?"):
        print(hit.score, hit.text)

    for event in tg.chat("summarize the corpus"):
        if event.event == "token":
            print(event.data["text"], end="")
"""

from .client import (
    Client,
    TurbographError,
    QueryResult,
    DocInfo,
    DocView,
    ChunkSpan,
    DocVersion,
    CommunitySummary,
    SSEEvent,
)

__all__ = [
    "Client",
    "TurbographError",
    "QueryResult",
    "DocInfo",
    "DocView",
    "ChunkSpan",
    "DocVersion",
    "CommunitySummary",
    "SSEEvent",
]

__version__ = "0.1.0"
