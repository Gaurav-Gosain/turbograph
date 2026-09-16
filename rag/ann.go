package rag

import "github.com/Gaurav-Gosain/turbograph/index"

// AnnIndex is the vector-search surface the store needs from its ANN backend.
// index.HNSW satisfies it. The seam exists so the store's hot paths depend on a
// small contract instead of a concrete type, and a different backend (a flat
// scan, an external service) can be swapped in without changing the store.
type AnnIndex interface {
	// Len returns the number of indexed vectors.
	Len() int
	// Ord resolves an id to its ordinal; ok is false for unknown ids.
	Ord(id string) (int, bool)
	// Vector returns the stored vector at an ordinal.
	Vector(ord int) []float32
	// Add inserts a vector under id and returns its ordinal.
	Add(id string, vec []float32) int
	// Search returns the top-k ids nearest to query at search width ef.
	Search(query []float32, k, ef int) []index.Result
	// SearchFiltered is Search restricted to ids for which accept returns true.
	SearchFiltered(query []float32, k, ef int, accept func(id string) bool) []index.Result
	// Snapshot returns the layered link graph, used to persist the index.
	Snapshot() index.Graph
}
