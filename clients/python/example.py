"""A runnable tour of the turbograph Python client.

Start a turbograph server first (defaults to http://localhost:8080), then run:

    python example.py

The script ingests a few documents, runs a query, streams a chat answer, and
prints document and status information. It degrades gracefully: steps that need a
language model are skipped with a note when no model is configured.
"""

from turbograph import Client, TurbographError


def main() -> None:
    client = Client("http://localhost:8080", bucket="example")

    # Make sure we have an isolated bucket to play in.
    try:
        client.create_bucket("example")
    except TurbographError:
        pass  # already exists

    print("== Ingesting text ==")
    res = client.ingest_text(
        [
            {"id": "fox", "text": "The quick brown fox jumps over the lazy dog."},
            {"id": "moon", "text": "The moon orbits the Earth roughly every 27 days."},
            {"id": "rome", "text": "Rome was founded, by legend, in 753 BC on the Palatine Hill."},
        ]
    )
    print(f"indexed, total chunks now: {res.get('chunks')}\n")

    print("== Query ==")
    for hit in client.query("what does the fox do?", top_k=3):
        print(f"  [{hit.score:.3f}] {hit.doc_id}: {hit.text}")
    print()

    print("== Documents ==")
    for doc in client.documents():
        print(f"  {doc.id}: {doc.chunks} chunk(s), {doc.bytes} bytes")
    print()

    print("== Status ==")
    status = client.status()
    print(f"  version: {status.get('version')}")
    gen = status.get("generation", {})
    print(f"  generation model: {gen.get('model')!r} reachable: {gen.get('reachable')}")
    print()

    # The chat endpoint needs a language model. Skip it cleanly if none is ready.
    if gen.get("reachable"):
        print("== Chat (streaming) ==")
        for event in client.chat("what does the fox do?", top_k=3):
            if event.event == "token":
                print(event.data.get("text", ""), end="", flush=True)
            elif event.event == "abstain":
                print(event.data.get("message", ""))
            elif event.event == "error":
                print(f"\n  error: {event.data.get('error')}")
        print("\n")

        print("== Chat (collected) ==")
        answer, sources = client.chat_text("how long is a lunar orbit?", top_k=3)
        print(f"  answer: {answer}")
        print(f"  sources: {[s.doc_id for s in sources]}")
    else:
        print("(no language model reachable; skipping chat)")


if __name__ == "__main__":
    try:
        main()
    except TurbographError as e:
        print(f"turbograph error: {e.message} (status {e.status})")
