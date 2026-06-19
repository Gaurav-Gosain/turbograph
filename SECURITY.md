# Security

## Reporting

Report vulnerabilities privately through GitHub's "Report a vulnerability"
(Security advisories) on the repository, or by opening a minimal issue asking for
a private contact if that is unavailable. Please do not disclose publicly until a
fix is available.

## Deployment notes

turbograph is local-first. When exposing `serve` beyond localhost:

- Set `--api-key` (or `$TURBOGRAPH_API_KEY`). Without it, every endpoint is open
  to anyone who can reach the port. The key is checked in constant time and
  accepted via `Authorization: Bearer`, `X-API-Key`, or `?api_key=`.
- Liveness (`/healthz`) and readiness (`/readyz`) are intentionally unauthenticated
  so orchestrators can probe a protected server.
- `--metrics` (`/debug/vars`) and `--pprof` (`/debug/pprof/`) are off by default
  and sit behind `--api-key` when one is set. The profiler can dump heap and CPU
  data, so only enable it with auth on, or on a non-public interface.
- Request bodies are capped (`--max-body`, default 32 MiB) and panics are
  recovered, but turbograph does no per-client rate limiting; put it behind a
  reverse proxy or gateway if you need that.
- The bundled `extract` command runners execute external programs you configure
  (`--pdf-cmd`, `--ocr-cmd`). Only configure tools you trust; uploaded bytes are
  passed to them via a temporary file.
- S3 credentials are read from the environment, never from flags or the store.

## Scope

The retrieval and generation paths send text to the configured Ollama server.
Nothing leaves the machine unless you point `--ollama-url` or `--s3-endpoint` at a
remote host.
