"""Unit tests for the turbograph client.

These tests need no live server. The SSE parser is tested directly, and the
HTTP-shaped methods are exercised by monkeypatching urllib.request.urlopen with a
fake that records the request and returns a canned response.
"""

import io
import json
import os
import sys
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbograph import Client, TurbographError, QueryResult  # noqa: E402
from turbograph.sse import parse_sse  # noqa: E402
import turbograph.client as client_module  # noqa: E402


# --------------------------------------------------------------------------- #
# SSE parser
# --------------------------------------------------------------------------- #


class TestSSEParser(unittest.TestCase):
    def test_basic_events(self):
        stream = (
            b'event: sources\ndata: {"sources": []}\n\n'
            b'event: token\ndata: {"text": "hi"}\n\n'
            b'event: done\ndata: {"done": true}\n\n'
        )
        events = list(parse_sse([stream]))
        self.assertEqual([e.event for e in events], ["sources", "token", "done"])
        self.assertEqual(events[1].data, {"text": "hi"})
        self.assertEqual(events[2].data, {"done": True})

    def test_chunk_boundaries_split_mid_event(self):
        # Feed the same stream one byte at a time; parsing must be identical.
        whole = b'event: token\ndata: {"text": "abc"}\n\nevent: done\ndata: {"done": true}\n\n'
        chunks = [whole[i : i + 1] for i in range(len(whole))]
        events = list(parse_sse(chunks))
        self.assertEqual([e.event for e in events], ["token", "done"])
        self.assertEqual(events[0].data, {"text": "abc"})

    def test_crlf_line_endings(self):
        stream = b'event: token\r\ndata: {"text": "x"}\r\n\r\n'
        events = list(parse_sse([stream]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "token")
        self.assertEqual(events[0].data, {"text": "x"})

    def test_default_event_name_is_message(self):
        events = list(parse_sse([b'data: {"a": 1}\n\n']))
        self.assertEqual(events[0].event, "message")
        self.assertEqual(events[0].data, {"a": 1})

    def test_multiple_data_lines_joined(self):
        events = list(parse_sse([b"event: x\ndata: line1\ndata: line2\n\n"]))
        self.assertEqual(events[0].raw, "line1\nline2")

    def test_comment_and_blank_lines_ignored(self):
        events = list(parse_sse([b':comment\nevent: ping\ndata: {"ok": true}\n\n']))
        self.assertEqual(events[0].event, "ping")
        self.assertEqual(events[0].data, {"ok": True})

    def test_trailing_event_without_blank_line(self):
        events = list(parse_sse([b'event: done\ndata: {"done": true}\n']))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "done")

    def test_non_json_data_kept_as_string(self):
        events = list(parse_sse([b"event: note\ndata: plain text\n\n"]))
        self.assertEqual(events[0].data, "plain text")


# --------------------------------------------------------------------------- #
# A fake urlopen for URL/body assertions without a server
# --------------------------------------------------------------------------- #


class FakeResponse(io.BytesIO):
    """Minimal stand-in for a urllib response object."""

    def __init__(self, body: bytes):
        super().__init__(body)


class RecordingURLOpen:
    """Records the last Request and returns a fixed JSON or SSE body."""

    def __init__(self, body: bytes):
        self.body = body
        self.last_request = None
        self.last_timeout = None

    def __call__(self, req, timeout=None):
        self.last_request = req
        self.last_timeout = timeout
        return FakeResponse(self.body)

    @property
    def url(self):
        return self.last_request.full_url

    @property
    def method(self):
        return self.last_request.get_method()

    @property
    def json_body(self):
        return json.loads(self.last_request.data.decode("utf-8"))


class ClientTestBase(unittest.TestCase):
    def install(self, body: bytes) -> RecordingURLOpen:
        fake = RecordingURLOpen(body)
        self._orig = client_module.urllib.request.urlopen
        client_module.urllib.request.urlopen = fake
        self.addCleanup(self._restore)
        return fake

    def _restore(self):
        client_module.urllib.request.urlopen = self._orig


# --------------------------------------------------------------------------- #
# URL and bucket building
# --------------------------------------------------------------------------- #


class TestURLBuilding(ClientTestBase):
    def test_base_url_trailing_slash_stripped(self):
        c = Client("http://localhost:8080/")
        self.assertEqual(c.base_url, "http://localhost:8080")

    def test_default_bucket_appended(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'{"results": []}')
        c.query("hello")
        self.assertIn("bucket=default", fake.url)
        self.assertTrue(fake.url.startswith("http://localhost:8080/query?"))
        self.assertEqual(fake.method, "POST")

    def test_custom_bucket_on_client(self):
        c = Client("http://localhost:8080", bucket="legal")
        fake = self.install(b'{"results": []}')
        c.query("hello")
        self.assertIn("bucket=legal", fake.url)

    def test_per_call_bucket_override(self):
        c = Client("http://localhost:8080", bucket="legal")
        fake = self.install(b'{"results": []}')
        c.query("hello", bucket="medical")
        self.assertIn("bucket=medical", fake.url)
        self.assertNotIn("bucket=legal", fake.url)

    def test_query_params_encoded(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'{"doc": "a", "text": "x", "spans": []}')
        c.document("a b/c")
        self.assertIn("doc=a+b%2Fc", fake.url)

    def test_models_and_buckets_have_no_bucket_param(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'{"models": []}')
        c.models()
        self.assertNotIn("bucket=", fake.url)

    def test_asset_url_helper(self):
        c = Client("http://localhost:8080")
        self.assertEqual(
            c.asset_url("ab12cd34ef56.png"),
            "http://localhost:8080/api/asset/ab12cd34ef56.png",
        )


# --------------------------------------------------------------------------- #
# Request bodies and response parsing
# --------------------------------------------------------------------------- #


class TestRequestBodies(ClientTestBase):
    def test_ingest_text_single_convenience(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'{"chunks": 1, "saved": true}')
        c.ingest_text("doc1", "some text")
        body = fake.json_body
        self.assertEqual(body["documents"], [{"id": "doc1", "text": "some text"}])
        self.assertFalse(body["replace"])

    def test_ingest_text_requires_text_for_single_id(self):
        c = Client("http://localhost:8080")
        with self.assertRaises(ValueError):
            c.ingest_text("doc1")

    def test_ingest_text_list_form(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'{"chunks": 2}')
        c.ingest_text([{"id": "a", "text": "x"}, {"id": "b", "text": "y"}], replace=True)
        body = fake.json_body
        self.assertEqual(len(body["documents"]), 2)
        self.assertTrue(body["replace"])

    def test_query_body_and_result_parsing(self):
        c = Client("http://localhost:8080")
        payload = {
            "results": [
                {
                    "id": "c1",
                    "doc_id": "d1",
                    "score": 0.9,
                    "similarity": 0.8,
                    "text": "hello",
                    "start": 0,
                    "end": 5,
                    "kind": "image",
                    "image_ref": "abc.png",
                }
            ]
        }
        fake = self.install(json.dumps(payload).encode())
        results = c.query("q", top_k=3, graph_mix=0.5)
        body = fake.json_body
        self.assertEqual(body["query"], "q")
        self.assertEqual(body["top_k"], 3)
        self.assertEqual(body["graph_mix"], 0.5)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIsInstance(r, QueryResult)
        self.assertEqual(r.id, "c1")
        self.assertEqual(r.image_ref, "abc.png")
        self.assertEqual(r.kind, "image")

    def test_chat_body_uses_global_key(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'event: done\ndata: {"done": true}\n\n')
        list(c.chat("q", global_=True, meta_keys=["title"], model="llama3"))
        body = fake.json_body
        self.assertTrue(body["global"])
        self.assertNotIn("global_", body)
        self.assertEqual(body["meta_keys"], ["title"])
        self.assertEqual(body["model"], "llama3")

    def test_ingest_image_encodes_bytes(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'{"id": "img1", "image_ref": "x.png", "caption": "a cat"}')
        out = c.ingest_image("img1", b"\x89PNG", "png", "llava")
        body = fake.json_body
        self.assertEqual(body["id"], "img1")
        self.assertEqual(body["ext"], "png")
        self.assertEqual(body["model"], "llava")
        # base64 of b"\x89PNG"
        import base64

        self.assertEqual(body["b64"], base64.b64encode(b"\x89PNG").decode())
        self.assertEqual(out["caption"], "a cat")

    def test_delete_document_uses_delete_method(self):
        c = Client("http://localhost:8080")
        fake = self.install(b'{"deleted": "d1", "chunks": 3}')
        c.delete_document("d1")
        self.assertEqual(fake.method, "DELETE")
        self.assertIn("doc=d1", fake.url)


# --------------------------------------------------------------------------- #
# Streaming convenience and build draining
# --------------------------------------------------------------------------- #


class TestStreamingConsumers(ClientTestBase):
    def test_chat_text_joins_tokens_and_collects_sources(self):
        c = Client("http://localhost:8080")
        body = (
            b'event: sources\ndata: {"sources": [{"id": "c1", "text": "ctx"}]}\n\n'
            b'event: token\ndata: {"text": "Hello "}\n\n'
            b'event: token\ndata: {"text": "world"}\n\n'
            b'event: done\ndata: {"done": true}\n\n'
        )
        self.install(body)
        answer, sources = c.chat_text("hi")
        self.assertEqual(answer, "Hello world")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].id, "c1")

    def test_chat_text_raises_on_error_event(self):
        c = Client("http://localhost:8080")
        self.install(b'event: error\ndata: {"error": "no model"}\n\n')
        with self.assertRaises(TurbographError) as ctx:
            c.chat_text("hi")
        self.assertEqual(ctx.exception.message, "no model")

    def test_chat_text_abstain_returns_message(self):
        c = Client("http://localhost:8080")
        body = (
            b'event: abstain\ndata: {"message": "nothing relevant"}\n\n'
            b'event: done\ndata: {"done": true}\n\n'
        )
        self.install(body)
        answer, sources = c.chat_text("hi")
        self.assertEqual(answer, "nothing relevant")
        self.assertEqual(sources, [])

    def test_build_communities_blocking_returns_done_payload(self):
        c = Client("http://localhost:8080")
        body = (
            b'event: progress\ndata: {"done": 1, "total": 2}\n\n'
            b'event: progress\ndata: {"done": 2, "total": 2}\n\n'
            b'event: done\ndata: {"communities": 2}\n\n'
        )
        self.install(body)
        out = c.build_communities_blocking("llama3")
        self.assertEqual(out, {"communities": 2})

    def test_build_entities_blocking_raises_on_error(self):
        c = Client("http://localhost:8080")
        self.install(b'event: error\ndata: {"error": "boom"}\n\n')
        with self.assertRaises(TurbographError):
            c.build_entities_blocking("llama3")


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


class TestErrorHandling(ClientTestBase):
    def test_http_error_extracts_server_message(self):
        c = Client("http://localhost:8080")

        def raising(req, timeout=None):
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error": "query is required"}'),
            )

        self._orig = client_module.urllib.request.urlopen
        client_module.urllib.request.urlopen = raising
        self.addCleanup(self._restore)

        with self.assertRaises(TurbographError) as ctx:
            c.query("")
        self.assertEqual(ctx.exception.message, "query is required")
        self.assertEqual(ctx.exception.status, 400)

    def test_url_error_wrapped(self):
        c = Client("http://localhost:8080")

        def raising(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        self._orig = client_module.urllib.request.urlopen
        client_module.urllib.request.urlopen = raising
        self.addCleanup(self._restore)

        with self.assertRaises(TurbographError) as ctx:
            c.status()
        self.assertIn("connection refused", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
