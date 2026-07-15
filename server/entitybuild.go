package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// entBuildJob is one entity-graph extraction, detached from the request that started
// it so it keeps running when that client reloads or navigates away. Every progress
// event is kept in order, so a client that (re)connects replays them and rebuilds the
// graph-so-far instead of staring at an empty canvas; the terminal result is held
// until the next build for the same bucket replaces the job.
type entBuildJob struct {
	mu       sync.Mutex
	events   []map[string]any // progress payloads, in order, for replay
	running  bool
	finished bool
	done     map[string]int // the terminal "done" payload
	errMsg   string
	// Snapshot counters, so the cheap status endpoint does not have to walk events.
	nDone, total, entities, relations int
	cancel                            context.CancelFunc
}

func (j *entBuildJob) record(p rag.EntityProgress) {
	// The payload mirrors what the old inline handler streamed, so the client's
	// progress handling is unchanged: "new" and "edges" carry the entities and
	// relationships this chunk surfaced, so the graph draws as it is discovered.
	payload := map[string]any{
		"done": p.Done, "total": p.Total, "cached": p.Cached,
		"entities": p.Entities, "relations": p.Relations,
		"new": p.New, "edges": p.NewRelations,
	}
	j.mu.Lock()
	j.events = append(j.events, payload)
	j.nDone, j.total, j.entities, j.relations = p.Done, p.Total, p.Entities, p.Relations
	j.mu.Unlock()
}

func (j *entBuildJob) fail(msg string) {
	j.mu.Lock()
	j.errMsg = msg
	j.running, j.finished = false, true
	j.mu.Unlock()
}

func (j *entBuildJob) complete(done map[string]int) {
	j.mu.Lock()
	j.done = done
	j.running, j.finished = false, true
	j.mu.Unlock()
}

// startEntBuild launches a detached build for the bucket unless one is already
// running, and returns the job to stream. It never starts a second concurrent build
// for the same bucket: a caller that arrives while one is in flight attaches to it.
func (s *Server) startEntBuild(bucket string, st *rag.Store, model string, batch int, refresh bool) *entBuildJob {
	s.entJobsMu.Lock()
	if s.entJobs == nil {
		s.entJobs = map[string]*entBuildJob{}
	}
	if j := s.entJobs[bucket]; j != nil {
		j.mu.Lock()
		running := j.running
		j.mu.Unlock()
		if running {
			s.entJobsMu.Unlock()
			return j
		}
	}
	job := &entBuildJob{running: true}
	s.entJobs[bucket] = job
	s.entJobsMu.Unlock()

	go s.runEntBuild(bucket, job, st, model, batch, refresh)
	return job
}

// entJob returns the current job for a bucket, or nil.
func (s *Server) entJob(bucket string) *entBuildJob {
	s.entJobsMu.Lock()
	defer s.entJobsMu.Unlock()
	return s.entJobs[bucket]
}

func (s *Server) runEntBuild(bucket string, job *entBuildJob, st *rag.Store, model string, batch int, refresh bool) {
	ctx, cancel := context.WithCancel(context.Background())
	job.mu.Lock()
	job.cancel = cancel
	job.mu.Unlock()
	defer cancel()

	var extracted, cached int
	ex := entity.NewLLMExtractor(genAdapter{c: s.gen, model: model})
	err := st.BuildEntityGraph(ctx, ex, rag.EntityBuildOptions{
		BatchSize: batch,
		Model:     model,
		Refresh:   refresh,
		OnProgress: func(p rag.EntityProgress) {
			extracted = p.Entities // raw count, before canonicalization prunes and merges
			cached = p.Cached
			job.record(p)
		},
	})
	if err != nil {
		job.fail(err.Error())
		return
	}
	s.persist(bucket)
	job.complete(map[string]int{
		"entities": st.EntityCount(), "extracted": extracted,
		"relations": st.RelationCount(), "cached": cached,
	})
}

// stream replays the job's progress so far and then follows it live over server-sent
// events, ending with the terminal "done" or "error". It returns when the job finishes
// or the client disconnects; a disconnect does not stop the build.
func (j *entBuildJob) stream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	send := func(event string, v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		flusher.Flush()
	}
	cursor := 0
	for {
		j.mu.Lock()
		batch := append([]map[string]any(nil), j.events[cursor:]...)
		cursor = len(j.events)
		fin, errMsg, done := j.finished, j.errMsg, j.done
		j.mu.Unlock()

		for _, e := range batch {
			send("progress", e)
		}
		if fin {
			if errMsg != "" {
				send("error", map[string]string{"error": errMsg})
			} else {
				send("done", done)
			}
			return
		}
		select {
		case <-r.Context().Done():
			return // the client left; the build goroutine keeps running
		case <-time.After(150 * time.Millisecond):
		}
	}
}
