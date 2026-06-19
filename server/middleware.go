package server

import (
	"crypto/subtle"
	"expvar"
	"fmt"
	"log"
	"net/http"
	"runtime/debug"
	"strings"
	"time"
)

// Options configures the hardening middleware around the HTTP handler. The zero
// value is safe and is what Handler() uses: sensible body limits, no auth, no
// CORS, no metrics endpoint. Everything here is standard-library only.
type Options struct {
	// MaxBodyBytes caps request body size. 0 selects DefaultMaxBodyBytes; a
	// negative value disables the limit (not recommended for public deployments).
	MaxBodyBytes int64
	// APIKey, if set, requires every request except liveness/readiness to present
	// it as "Authorization: Bearer <key>", an "X-API-Key" header, or an "api_key"
	// query parameter. The comparison is constant time.
	APIKey string
	// CORSOrigin, if set, is echoed in Access-Control-Allow-Origin and enables
	// preflight handling. Use "*" to allow any origin.
	CORSOrigin string
	// Metrics exposes process and request counters at /debug/vars (expvar).
	Metrics bool
	// Version is reported by the health endpoint so a deployment can be confirmed
	// with a single probe. Empty defaults to "dev".
	Version string
}

// DefaultMaxBodyBytes bounds request bodies (uploads use a dedicated larger limit
// path). 32 MiB is generous for JSON and base64 file batches without inviting
// memory-exhaustion from a single request.
const DefaultMaxBodyBytes = 32 << 20

var (
	metricRequests = expvar.NewInt("turbograph_requests_total")
	metricErrors   = expvar.NewInt("turbograph_responses_5xx_total")
	metricInflight = expvar.NewInt("turbograph_requests_in_flight")
	startTime      = time.Now()
)

func init() {
	expvar.Publish("turbograph_uptime_seconds", expvar.Func(func() any {
		return int64(time.Since(startTime).Seconds())
	}))
}

// chain applies the hardening middleware in the right order around next:
// recover (outermost, so a panic anywhere becomes a 500), then metrics, CORS,
// auth, body limit, and finally the access log closest to the handler.
func chain(next http.Handler, opt Options) http.Handler {
	limit := opt.MaxBodyBytes
	if limit == 0 {
		limit = DefaultMaxBodyBytes
	}
	h := logging(next)
	h = bodyLimit(h, limit)
	h = auth(h, opt.APIKey)
	h = cors(h, opt.CORSOrigin)
	if opt.Metrics {
		h = withMetrics(h)
	}
	h = recoverPanic(h)
	return h
}

// recoverPanic turns a panic in any handler into a logged 500 instead of
// crashing the process, the single most important guard for a long-running
// daemon. It writes a 500 only if the handler had not started the response.
func recoverPanic(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sw := asStatus(w)
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("panic serving %s %s: %v\n%s", r.Method, r.URL.Path, rec, debug.Stack())
				if sw.status == 0 {
					writeErr(sw, http.StatusInternalServerError, fmt.Errorf("internal error"))
				}
			}
		}()
		next.ServeHTTP(sw, r)
	})
}

// bodyLimit caps the request body so a single client cannot exhaust memory. The
// streaming UI and uploads still work: the limit is per request, not per stream.
func bodyLimit(next http.Handler, max int64) http.Handler {
	if max < 0 {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, max)
		}
		next.ServeHTTP(w, r)
	})
}

// auth enforces an API key when one is configured. Liveness and readiness stay
// open so orchestrators can probe an otherwise-protected server.
func auth(next http.Handler, key string) http.Handler {
	if key == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
			next.ServeHTTP(w, r)
			return
		}
		if r.Method == http.MethodOptions { // let CORS preflight through
			next.ServeHTTP(w, r)
			return
		}
		if !validKey(r, key) {
			w.Header().Set("WWW-Authenticate", "Bearer")
			writeErr(w, http.StatusUnauthorized, fmt.Errorf("missing or invalid API key"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func validKey(r *http.Request, key string) bool {
	got := r.Header.Get("X-API-Key")
	if got == "" {
		if b := r.Header.Get("Authorization"); strings.HasPrefix(b, "Bearer ") {
			got = strings.TrimPrefix(b, "Bearer ")
		}
	}
	if got == "" {
		got = r.URL.Query().Get("api_key")
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(key)) == 1
}

// cors adds permissive-but-explicit CORS headers and answers preflight requests
// when an origin is configured, so browser clients on another origin can call the
// API and the OpenAI-compatible endpoint.
func cors(next http.Handler, origin string) http.Handler {
	if origin == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
		w.Header().Set("Vary", "Origin")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// withMetrics counts requests, in-flight requests, and 5xx responses into expvar
// so /debug/vars exposes basic operational telemetry with no dependency.
func withMetrics(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		metricRequests.Add(1)
		metricInflight.Add(1)
		defer metricInflight.Add(-1)
		sw := asStatus(w)
		next.ServeHTTP(sw, r)
		if sw.status >= 500 {
			metricErrors.Add(1)
		}
	})
}
