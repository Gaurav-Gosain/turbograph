package rag

import "context"

// embedBatched embeds texts in bounded batches of Config.EmbedBatchSize,
// preserving order. It is the single seam between the store and its embedder
// for document text, so a large document or corpus is never one oversized
// request (some backends' tokenizers choke on a huge single batch). Empty
// input returns without calling the embedder. It does not touch store state,
// so it needs no lock; every caller embeds off the write lock.
func (s *Store) embedBatched(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	batch := s.cfg.EmbedBatchSize
	if batch < 1 {
		batch = 64
	}
	if len(texts) <= batch {
		return s.embedder.Embed(ctx, texts)
	}
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += batch {
		end := min(i+batch, len(texts))
		vecs, err := s.embedder.Embed(ctx, texts[i:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}
