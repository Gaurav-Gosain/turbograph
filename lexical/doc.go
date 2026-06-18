// Package lexical implements sparse lexical retrieval to complement dense
// vector search in a hybrid retrieval-augmented generation (RAG) pipeline.
//
// Dense retrieval (embeddings + nearest neighbour search) excels at semantic
// matching: it finds passages that mean the same thing as a query even when
// they share no words. Its weakness is the inverse. Rare but decisive tokens,
// identifiers, product codes, names, and exact phrases are often smeared away
// in the embedding space, so a dense index can miss a document that literally
// contains the user's search term. Lexical retrieval covers exactly that gap.
// An Okapi BM25 index ranks documents by term overlap weighted toward rare
// terms and normalized for document length, giving strong precision on exact
// keyword matches.
//
// Neither signal dominates across all queries, so production RAG systems run
// both and fuse the results. This package provides BM25 for the sparse half
// and Reciprocal Rank Fusion (RRF) for combining ranked lists from independent
// retrievers. RRF works on ranks rather than raw scores, which sidesteps the
// problem that BM25 scores and cosine similarities live on incomparable scales
// and need no per-query calibration.
package lexical
