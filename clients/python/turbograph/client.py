"""The turbograph HTTP client.

A thin, typed wrapper over turbograph's HTTP+JSON API and its server-sent-events
streaming endpoints, built on the Python standard library only.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .sse import SSEEvent, parse_sse

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


class TurbographError(Exception):
    """Raised when the server returns a non-2xx response.

    Attributes:
        message: the human-readable error, taken from the server's
            {"error": "..."} body when present, else the raw body or HTTP reason.
        status: the HTTP status code, or None for a transport-level failure.
        body: the raw response body, when available.
    """

    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.body = body


# Dataclasses mirroring the server's JSON response shapes. Each from_dict tolerates
# missing optional fields, since the server omits empty/zero values in several places.


@dataclass
class QueryResult:
    """One retrieved chunk, as returned by /query and as a chat "sources" entry.

    Mirrors the server's queryResult struct.
    """

    id: str = ""
    doc_id: str = ""
    score: float = 0.0
    similarity: float = 0.0
    text: str = ""
    start: int = 0
    end: int = 0
    meta: Optional[Dict[str, Any]] = None
    kind: str = ""
    image_ref: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QueryResult":
        return cls(
            id=d.get("id", ""),
            doc_id=d.get("doc_id", ""),
            score=d.get("score", 0.0),
            similarity=d.get("similarity", 0.0),
            text=d.get("text", ""),
            start=d.get("start", 0),
            end=d.get("end", 0),
            meta=d.get("meta"),
            kind=d.get("kind", ""),
            image_ref=d.get("image_ref", ""),
        )


@dataclass
class DocInfo:
    """A document listing entry from /api/documents (the server's DocInfo)."""

    id: str = ""
    chunks: int = 0
    bytes: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocInfo":
        return cls(id=d.get("id", ""), chunks=d.get("chunks", 0), bytes=d.get("bytes", 0))


@dataclass
class ChunkSpan:
    """The position of one chunk within its document (the server's ChunkSpan)."""

    id: str = ""
    pos: int = 0
    start: int = 0
    end: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkSpan":
        return cls(
            id=d.get("id", ""),
            pos=d.get("pos", 0),
            start=d.get("start", 0),
            end=d.get("end", 0),
        )


@dataclass
class DocView:
    """A full document with its metadata and chunk spans (the server's DocView)."""

    id: str = ""
    text: str = ""
    meta: Optional[Dict[str, Any]] = None
    spans: List[ChunkSpan] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocView":
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            meta=d.get("meta"),
            spans=[ChunkSpan.from_dict(s) for s in d.get("spans") or []],
        )


@dataclass
class DocVersion:
    """One entry in a document's version history (the server's DocVersion)."""

    n: int = 0
    hash: str = ""
    time: int = 0
    bytes: int = 0
    chunks: int = 0
    current: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocVersion":
        return cls(
            n=d.get("n", 0),
            hash=d.get("hash", ""),
            time=d.get("time", 0),
            bytes=d.get("bytes", 0),
            chunks=d.get("chunks", 0),
            current=d.get("current", False),
        )


@dataclass
class CommunitySummary:
    """A thematic community summary from /api/communities (the server's CommunitySummary)."""

    label: int = 0
    size: int = 0
    summary: str = ""
    chunks: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommunitySummary":
        return cls(
            label=d.get("label", 0),
            size=d.get("size", 0),
            summary=d.get("summary", ""),
            chunks=list(d.get("chunks") or []),
            doc_ids=list(d.get("doc_ids") or []),
        )


# Map common file extensions to whether the bytes are textual. Text files can be
# ingested directly through /ingest; everything else needs server-side extraction
# (for example PDF) through /api/ingest/files.
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".text", ".log",
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml",
    ".xml", ".html", ".htm", ".css", ".tex",
    ".py", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".kt",
    ".js", ".jsx", ".ts", ".tsx", ".rb", ".php", ".sh", ".bash", ".zsh",
    ".sql", ".ini", ".cfg", ".conf", ".env",
}


def _is_text_ext(ext: str) -> bool:
    """Report whether a file extension denotes a plain-text format."""
    return ext.lower() in _TEXT_EXTENSIONS


class Client:
    """A client for a turbograph server.

    All bucket-scoped calls append a ``bucket`` query parameter; the bucket is
    chosen at construction and can be overridden per call.

    Args:
        base_url: the server root, for example "http://localhost:8080".
        bucket: the corpus to operate on; defaults to "default".
        timeout: socket timeout in seconds for non-streaming requests.
        headers: extra headers sent with every request (for auth, say).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        bucket: str = "default",
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.bucket = bucket
        self.timeout = timeout
        self.headers = dict(headers or {})

    # ------------------------------------------------------------------ #
    # URL and request plumbing
    # ------------------------------------------------------------------ #

    def _url(self, path: str, params: Optional[Dict[str, Any]] = None, bucket: Optional[str] = None) -> str:
        """Build a full URL for an API path, appending query params.

        When ``bucket`` is given (or defaults to the client's bucket), a
        ``bucket`` query parameter is added unless one is already present.
        """
        query: Dict[str, Any] = {}
        if bucket is not None:
            query["bucket"] = bucket
        if params:
            for k, v in params.items():
                if v is not None:
                    query[k] = v
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        bucket: Optional[str] = None,
        stream: bool = False,
    ):
        """Issue a request and return the open response (caller closes it).

        Args:
            method: HTTP method.
            path: API path beginning with "/".
            params: query parameters.
            body: a JSON-serializable object sent as the request body.
            bucket: the bucket to scope to; None means do not add a bucket param.
            stream: when True, do not read or decode the body here.

        Returns:
            The urllib response object, left open for the caller.

        Raises:
            TurbographError: on a non-2xx status or transport failure.
        """
        url = self._url(path, params, bucket)
        data = None
        headers = {"Accept": "application/json", **self.headers}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if stream:
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        # Streaming responses must not use a fixed timeout: the connection stays
        # open for the life of the generation. Non-streaming calls use self.timeout.
        timeout = None if stream else self.timeout
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise self._error_from_http(e) from None
        except urllib.error.URLError as e:
            raise TurbographError(f"request to {url} failed: {e.reason}") from None

    @staticmethod
    def _error_from_http(e: "urllib.error.HTTPError") -> TurbographError:
        """Turn an HTTPError into a TurbographError, extracting {"error": ...}."""
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        message = body
        if body:
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and "error" in parsed:
                    message = parsed["error"]
            except ValueError:
                pass
        if not message:
            message = getattr(e, "reason", None) or f"HTTP {e.code}"
        return TurbographError(message, status=e.code, body=body)

    def _json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        bucket: Optional[str] = None,
    ) -> Any:
        """Issue a request and decode the JSON response body."""
        resp = self._request(method, path, params=params, body=body, bucket=bucket)
        try:
            raw = resp.read()
        finally:
            resp.close()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _stream(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        bucket: Optional[str] = None,
        chunk_size: int = 1024,
    ) -> Iterator[SSEEvent]:
        """Issue a request and yield parsed SSE events from the response."""
        resp = self._request(method, path, params=params, body=body, bucket=bucket, stream=True)

        def chunks() -> Iterator[bytes]:
            try:
                while True:
                    block = resp.read(chunk_size)
                    if not block:
                        break
                    yield block
            finally:
                resp.close()

        return parse_sse(chunks())

    def _bucket(self, bucket: Optional[str]) -> str:
        return bucket if bucket is not None else self.bucket

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def ingest_text(
        self,
        documents: Union[List[Dict[str, Any]], str],
        text: Optional[str] = None,
        *,
        replace: bool = False,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest one or more plain-text documents (POST /ingest).

        Two calling styles are supported::

            client.ingest_text([{"id": "a", "text": "..."}, {"id": "b", "text": "..."}])
            client.ingest_text("a", "the text of document a")

        Each document is a mapping with "id" and "text", and an optional "meta"
        mapping of arbitrary metadata.

        Args:
            documents: a list of document dicts, or a single document id when
                ``text`` is also given.
            text: the text for the single-document convenience form.
            replace: when True, rebuild the bucket from scratch from these
                documents; otherwise add incrementally.
            bucket: override the client's bucket for this call.

        Returns:
            The server's response, including "chunks" and "saved".
        """
        if isinstance(documents, str):
            if text is None:
                raise ValueError("text is required when documents is a single id")
            docs = [{"id": documents, "text": text}]
        else:
            docs = list(documents)
        payload = {"documents": docs, "replace": replace}
        return self._json("POST", "/ingest", body=payload, bucket=self._bucket(bucket))

    def ingest_files(
        self,
        files: List[Dict[str, Any]],
        *,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest binary files for server-side text extraction (POST /api/ingest/files).

        Each file is a mapping with "id" and "b64" (the base64-encoded bytes), and
        an optional "meta" mapping. The server extracts text (for example, PDF)
        and indexes it. Files that fail to parse appear in the response "failed".

        Args:
            files: a list of file dicts {"id", "b64", "meta"?}.
            bucket: override the client's bucket for this call.

        Returns:
            The server's response: "chunks", "indexed", "failed", "saved".
        """
        return self._json("POST", "/api/ingest/files", body={"files": files}, bucket=self._bucket(bucket))

    def ingest_path(
        self,
        path: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        doc_id: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a file from disk, choosing the right endpoint by extension.

        Text files (by extension) are read and sent through /ingest directly;
        everything else is base64-encoded and sent through /api/ingest/files for
        server-side extraction.

        Args:
            path: the file path on the local disk.
            meta: optional metadata to attach to the document.
            doc_id: the document id; defaults to the file's base name.
            bucket: override the client's bucket for this call.

        Returns:
            The relevant endpoint's response.
        """
        ident = doc_id or os.path.basename(path)
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as fh:
            data = fh.read()
        if _is_text_ext(ext):
            text = data.decode("utf-8", errors="replace")
            doc: Dict[str, Any] = {"id": ident, "text": text}
            if meta:
                doc["meta"] = meta
            return self.ingest_text([doc], bucket=bucket)
        b64 = base64.b64encode(data).decode("ascii")
        entry: Dict[str, Any] = {"id": ident, "b64": b64}
        if meta:
            entry["meta"] = meta
        return self.ingest_files([entry], bucket=bucket)

    def ingest_image(
        self,
        id: str,
        image: bytes,
        ext: str,
        model: str,
        prompt: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        *,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest an image: store it, caption it with a vision model, index the caption.

        The image is captioned by ``model`` and the caption is indexed as an image
        chunk that references the stored asset, so the image retrieves by its
        description in the same hybrid search as text (POST /api/ingest/image).

        Args:
            id: the document id for the image.
            image: the raw image bytes.
            ext: the file extension, for example "png" or "jpg".
            model: the vision model to caption with.
            prompt: an optional captioning instruction.
            meta: optional document metadata.
            bucket: override the client's bucket for this call.

        Returns:
            The server's response: "id", "image_ref", "caption".
        """
        payload: Dict[str, Any] = {
            "id": id,
            "b64": base64.b64encode(image).decode("ascii"),
            "ext": ext.lstrip("."),
            "model": model,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if meta is not None:
            payload["meta"] = meta
        return self._json("POST", "/api/ingest/image", body=payload, bucket=self._bucket(bucket))

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def query(
        self,
        text: str,
        top_k: int = 6,
        graph_mix: float = 0.0,
        mmr_lambda: float = 0.0,
        entity_mix: float = 0.0,
        *,
        bucket: Optional[str] = None,
    ) -> List[QueryResult]:
        """Retrieve the chunks most relevant to a query (POST /query).

        Args:
            text: the query text.
            top_k: number of results to return.
            graph_mix: weight of graph-based reranking, in [0, 1].
            mmr_lambda: maximal-marginal-relevance diversity weight, in [0, 1].
            entity_mix: weight of entity-graph signal, in [0, 1].
            bucket: override the client's bucket for this call.

        Returns:
            A list of QueryResult, best first.
        """
        payload = {
            "query": text,
            "top_k": top_k,
            "graph_mix": graph_mix,
            "mmr_lambda": mmr_lambda,
            "entity_mix": entity_mix,
        }
        resp = self._json("POST", "/query", body=payload, bucket=self._bucket(bucket))
        return [QueryResult.from_dict(r) for r in resp.get("results") or []]

    # ------------------------------------------------------------------ #
    # Chat (streaming)
    # ------------------------------------------------------------------ #

    def chat(
        self,
        query: str,
        top_k: int = 6,
        graph_mix: float = 0.0,
        mmr_lambda: float = 0.0,
        entity_mix: float = 0.0,
        min_sim: float = 0.0,
        rerank: bool = False,
        global_: bool = False,
        meta_keys: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        *,
        bucket: Optional[str] = None,
    ) -> Iterator[SSEEvent]:
        """Stream a retrieval-augmented chat answer (POST /api/chat).

        This is a generator over the server's SSE stream. The events, in order,
        are typically:

        - "sources": data {"sources": [QueryResult-shaped dicts]}
        - "token":   data {"text": "..."} for each generated token (repeated)
        - "abstain": data {"message": "..."} when the evidence gate fires
        - "error":   data {"error": "..."} on failure
        - "done":    data {"done": true} at the end

        Args:
            query: the user's question.
            top_k: number of passages to retrieve.
            graph_mix, mmr_lambda, entity_mix: retrieval tuning (see ``query``).
            min_sim: abstain if the top hit's cosine similarity is below this.
            rerank: enable pointwise LLM reranking of candidates.
            global_: answer corpus-wide from community summaries instead of
                individual passages (requires built community summaries).
            meta_keys: document metadata keys to include in each passage.
            history: prior turns as [{"role": ..., "content": ...}], used to
                rewrite an elliptical follow-up into a standalone query.
            model: the generation model; defaults to the server's default.
            bucket: override the client's bucket for this call.

        Yields:
            SSEEvent objects as the server produces them.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "graph_mix": graph_mix,
            "mmr_lambda": mmr_lambda,
            "entity_mix": entity_mix,
            "min_sim": min_sim,
            "rerank": rerank,
            "global": global_,
        }
        if meta_keys is not None:
            payload["meta_keys"] = meta_keys
        if history is not None:
            payload["history"] = history
        if model is not None:
            payload["model"] = model
        return self._stream("POST", "/api/chat", body=payload, bucket=self._bucket(bucket))

    def chat_text(self, query: str, **kwargs: Any) -> Tuple[str, List[QueryResult]]:
        """Consume a chat stream and return the full answer plus its sources.

        A convenience over :meth:`chat` that joins all "token" events into one
        string and collects the "sources" event. Accepts the same keyword
        arguments as :meth:`chat`.

        Returns:
            A (answer, sources) tuple. ``answer`` is the abstain message when the
            server abstains.

        Raises:
            TurbographError: if the stream emits an "error" event.
        """
        parts: List[str] = []
        sources: List[QueryResult] = []
        for event in self.chat(query, **kwargs):
            if event.event == "token":
                parts.append((event.data or {}).get("text", ""))
            elif event.event == "sources":
                sources = [QueryResult.from_dict(s) for s in (event.data or {}).get("sources") or []]
            elif event.event == "abstain":
                parts.append((event.data or {}).get("message", ""))
            elif event.event == "error":
                raise TurbographError((event.data or {}).get("error", "chat error"))
            elif event.event == "done":
                break
        return "".join(parts), sources

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #

    def documents(self, *, bucket: Optional[str] = None) -> List[DocInfo]:
        """List the documents in the bucket (GET /api/documents)."""
        resp = self._json("GET", "/api/documents", bucket=self._bucket(bucket))
        return [DocInfo.from_dict(d) for d in resp.get("documents") or []]

    def document(self, doc: str, *, bucket: Optional[str] = None) -> DocView:
        """Fetch a document's full text, metadata, and chunk spans (GET /api/document).

        Args:
            doc: the document id.
            bucket: override the client's bucket for this call.
        """
        resp = self._json("GET", "/api/document", params={"doc": doc}, bucket=self._bucket(bucket))
        return DocView.from_dict(resp)

    def delete_document(self, doc: str, *, bucket: Optional[str] = None) -> Dict[str, Any]:
        """Delete a document and its chunks (DELETE /api/document).

        Returns:
            The server's response: "deleted" and the number of "chunks" removed.
        """
        return self._json("DELETE", "/api/document", params={"doc": doc}, bucket=self._bucket(bucket))

    # ------------------------------------------------------------------ #
    # Versions
    # ------------------------------------------------------------------ #

    def versions(self, doc: str, *, bucket: Optional[str] = None) -> List[DocVersion]:
        """List a document's version history, oldest first (GET /api/versions)."""
        resp = self._json("GET", "/api/versions", params={"doc": doc}, bucket=self._bucket(bucket))
        return [DocVersion.from_dict(v) for v in resp.get("versions") or []]

    def version_text(self, doc: str, n: int, *, bucket: Optional[str] = None) -> str:
        """Fetch the stored text of one document version (GET /api/version).

        Args:
            doc: the document id.
            n: the 1-based version number (oldest is 1).
            bucket: override the client's bucket for this call.
        """
        resp = self._json("GET", "/api/version", params={"doc": doc, "n": n}, bucket=self._bucket(bucket))
        return resp.get("text", "")

    def restore(self, doc: str, n: int, *, bucket: Optional[str] = None) -> Dict[str, Any]:
        """Restore an earlier document version (POST /api/restore).

        Re-ingests the stored text of version ``n``, appending it as a new
        version (git-revert semantics).

        Returns:
            The server's response: "doc", "restored", and the updated "versions".
        """
        return self._json("POST", "/api/restore", params={"doc": doc, "n": n}, bucket=self._bucket(bucket))

    # ------------------------------------------------------------------ #
    # Entity and community graph (streaming builds)
    # ------------------------------------------------------------------ #

    def build_entities(
        self,
        model: Optional[str] = None,
        batch: int = 4,
        *,
        bucket: Optional[str] = None,
    ) -> Iterator[SSEEvent]:
        """Build the entity-relationship graph, streaming progress (POST /api/build-entities).

        This is the GraphRAG-style indexing pass; it is expensive and on demand.
        Events: "progress" {"done","total","entities","relations"}, then "done"
        {"entities": n} or "error" {"error": ...}.

        Args:
            model: the generation model; defaults to the server's default.
            batch: chunks per model call (1 maximizes small-model fidelity).
            bucket: override the client's bucket for this call.

        Yields:
            SSEEvent objects as the build progresses.
        """
        params: Dict[str, Any] = {"batch": batch}
        if model is not None:
            params["model"] = model
        return self._stream("POST", "/api/build-entities", params=params, bucket=self._bucket(bucket))

    def build_entities_blocking(
        self,
        model: Optional[str] = None,
        batch: int = 4,
        *,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the entity graph and block until it completes.

        Drains the SSE stream from :meth:`build_entities` and returns the payload
        of the final "done" event.

        Raises:
            TurbographError: if the stream emits an "error" event.
        """
        return self._drain_build(self.build_entities(model, batch, bucket=bucket))

    def build_communities(
        self,
        model: Optional[str] = None,
        *,
        max_passages: Optional[int] = None,
        bucket: Optional[str] = None,
    ) -> Iterator[SSEEvent]:
        """Build community summaries, streaming progress (POST /api/build-communities).

        Generates one thematic summary per community of the chunk similarity
        graph, powering global corpus-wide questions. Events: "progress"
        {"done","total"}, then "done" {"communities": n} or "error".

        Args:
            model: the generation model; defaults to the server's default.
            max_passages: cap member passages per community in the prompt.
            bucket: override the client's bucket for this call.

        Yields:
            SSEEvent objects as the build progresses.
        """
        params: Dict[str, Any] = {}
        if model is not None:
            params["model"] = model
        if max_passages is not None:
            params["max_passages"] = max_passages
        return self._stream("POST", "/api/build-communities", params=params, bucket=self._bucket(bucket))

    def build_communities_blocking(
        self,
        model: Optional[str] = None,
        *,
        max_passages: Optional[int] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build community summaries and block until they complete.

        Drains the SSE stream from :meth:`build_communities` and returns the
        payload of the final "done" event.

        Raises:
            TurbographError: if the stream emits an "error" event.
        """
        return self._drain_build(self.build_communities(model, max_passages=max_passages, bucket=bucket))

    @staticmethod
    def _drain_build(stream: Iterator[SSEEvent]) -> Dict[str, Any]:
        """Consume a build SSE stream, returning the "done" payload or raising on error."""
        last: Dict[str, Any] = {}
        for event in stream:
            if event.event == "error":
                raise TurbographError((event.data or {}).get("error", "build error"))
            if event.event == "done":
                return event.data if isinstance(event.data, dict) else {}
            if event.event == "progress" and isinstance(event.data, dict):
                last = event.data
        return last

    def communities(self, *, bucket: Optional[str] = None) -> List[CommunitySummary]:
        """List the generated community summaries (GET /api/communities)."""
        resp = self._json("GET", "/api/communities", bucket=self._bucket(bucket))
        return [CommunitySummary.from_dict(c) for c in resp.get("communities") or []]

    # ------------------------------------------------------------------ #
    # Server, models, persistence
    # ------------------------------------------------------------------ #

    def models(self) -> Dict[str, Any]:
        """List available generation models and capabilities (GET /api/models).

        Returns:
            The server's response: "models", "default", "pdf", "embed_model",
            "embed_ready".
        """
        return self._json("GET", "/api/models", bucket=None)

    def status(self, *, bucket: Optional[str] = None) -> Dict[str, Any]:
        """Fetch server status, backend readiness, and bucket stats (GET /api/status)."""
        return self._json("GET", "/api/status", bucket=self._bucket(bucket))

    def save(self, *, bucket: Optional[str] = None) -> Dict[str, Any]:
        """Persist the bucket to disk (POST /api/save).

        A no-op success on an in-memory server.

        Returns:
            The server's response: "saved", "bucket", "path".
        """
        return self._json("POST", "/api/save", bucket=self._bucket(bucket))

    # ------------------------------------------------------------------ #
    # Buckets
    # ------------------------------------------------------------------ #

    def buckets(self) -> Dict[str, Any]:
        """List buckets with basic stats (GET /api/buckets).

        Returns:
            The server's response: "buckets" (a list of {name, chunks, documents,
            communities}) and the "default" bucket name.
        """
        return self._json("GET", "/api/buckets", bucket=None)

    def create_bucket(self, name: str) -> Dict[str, Any]:
        """Create a bucket (POST /api/buckets).

        Returns:
            The server's response: {"created": name}.
        """
        return self._json("POST", "/api/buckets", bucket=name)

    def delete_bucket(self, name: str) -> Dict[str, Any]:
        """Delete a bucket (DELETE /api/buckets).

        The default bucket cannot be deleted.

        Returns:
            The server's response: {"deleted": name}.
        """
        return self._json("DELETE", "/api/buckets", bucket=name)

    # ------------------------------------------------------------------ #
    # Assets
    # ------------------------------------------------------------------ #

    def asset_url(self, image_ref: str) -> str:
        """Return the absolute URL of a stored image asset.

        Args:
            image_ref: the asset id (the ``image_ref`` of an image chunk).
        """
        return self.base_url + "/api/asset/" + urllib.parse.quote(image_ref, safe="")

    def get_asset(self, image_ref: str) -> bytes:
        """Download a stored image asset's bytes (GET /api/asset/{id}).

        Args:
            image_ref: the asset id (the ``image_ref`` of an image chunk).

        Returns:
            The raw image bytes.
        """
        resp = self._request("GET", "/api/asset/" + urllib.parse.quote(image_ref, safe=""), bucket=None)
        try:
            return resp.read()
        finally:
            resp.close()
