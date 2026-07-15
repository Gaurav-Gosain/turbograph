package server

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// A client that connects after a build has already made progress must replay the
// whole history and then receive the terminal event, so a reload rebuilds the
// graph-so-far rather than showing an empty canvas. This is the guarantee that lets
// the build survive a reload, tested without a model.
func TestEntBuildStreamReplaysHistoryThenDone(t *testing.T) {
	job := &entBuildJob{running: true}
	job.record(rag.EntityProgress{Done: 1, Total: 3, Entities: 2, Relations: 1})
	job.record(rag.EntityProgress{Done: 2, Total: 3, Entities: 5, Relations: 4})
	job.complete(map[string]int{"entities": 5, "relations": 4})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/build-entities", nil)
	done := make(chan struct{})
	go func() { job.stream(rec, req); close(done) }()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("stream did not return after the job finished")
	}

	out := rec.Body.String()
	if n := strings.Count(out, "event: progress"); n != 2 {
		t.Fatalf("expected 2 replayed progress events, got %d in:\n%s", n, out)
	}
	if !strings.Contains(out, "event: done") {
		t.Fatalf("expected a terminal done event, got:\n%s", out)
	}
	if !strings.Contains(out, `"done":2`) {
		t.Errorf("replay should include the latest progress payload, got:\n%s", out)
	}
}

// A finished build that failed streams an error, not a done, so a reconnecting client
// surfaces the failure instead of hanging.
func TestEntBuildStreamReportsError(t *testing.T) {
	job := &entBuildJob{running: true}
	job.record(rag.EntityProgress{Done: 1, Total: 2})
	job.fail("model unreachable")

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/build-entities", nil)
	job.stream(rec, req)

	out := rec.Body.String()
	if !strings.Contains(out, "event: error") || !strings.Contains(out, "model unreachable") {
		t.Fatalf("expected an error event, got:\n%s", out)
	}
	if strings.Contains(out, "event: done") {
		t.Errorf("a failed build must not send done, got:\n%s", out)
	}
}

// A reconnect that hits a client-side cancel returns instead of blocking forever, and
// the build (a separate goroutine) is unaffected.
func TestEntBuildStreamReturnsOnClientCancel(t *testing.T) {
	job := &entBuildJob{running: true} // never finishes on its own
	job.record(rag.EntityProgress{Done: 1, Total: 10})

	ctx, cancel := context.WithCancel(context.Background())
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/build-entities", nil).WithContext(ctx)

	done := make(chan struct{})
	go func() { job.stream(rec, req); close(done) }()
	cancel()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("stream did not return when the client disconnected")
	}
}
