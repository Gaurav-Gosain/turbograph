package server

import (
	"context"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// fakeVerifyGen returns a canned generation, enough to exercise the audit parser.
type fakeVerifyGen struct{ resp string }

func (f fakeVerifyGen) Generate(context.Context, string, string, string) (string, error) {
	return f.resp, nil
}
func (f fakeVerifyGen) GenerateStream(context.Context, string, string, string, func(string) error) error {
	return nil
}
func (f fakeVerifyGen) ListModels(context.Context) ([]string, error) { return nil, nil }
func (f fakeVerifyGen) Ping(context.Context) error                   { return nil }

func TestAuditFaithfulness(t *testing.T) {
	s := New(rag.New(hashEmbedder{dim: 32}, rag.Config{Seed: 1}))
	// Model returns verdicts for claims 1 and 3, and skips claim 2.
	s.SetGenerator(fakeVerifyGen{resp: "1: SUPPORTED\n3: UNSUPPORTED\n"}, "m", "e")

	answer := "The reactor draws forty megawatts during a run. It was commissioned in 2019. The cooling loop uses argon gas."
	res := []rag.Retrieved{{Chunk: rag.Chunk{ID: "a#0", Text: "The reactor draws forty megawatts."}}}

	verdicts := s.auditFaithfulness(context.Background(), "m", answer, res)
	if len(verdicts) != 3 {
		t.Fatalf("expected 3 claim verdicts, got %d: %+v", len(verdicts), verdicts)
	}
	if verdicts[0].Verdict != "supported" {
		t.Errorf("claim 1 should be supported, got %q", verdicts[0].Verdict)
	}
	// Claim 2 was skipped by the model, so it defaults to partial (never a
	// confident pass/fail).
	if verdicts[1].Verdict != "partial" {
		t.Errorf("skipped claim should default to partial, got %q", verdicts[1].Verdict)
	}
	if verdicts[2].Verdict != "unsupported" {
		t.Errorf("claim 3 should be unsupported, got %q", verdicts[2].Verdict)
	}

	// No evidence -> no audit.
	if got := s.auditFaithfulness(context.Background(), "m", answer, nil); got != nil {
		t.Errorf("expected nil audit with no evidence, got %+v", got)
	}
	// Trivial fragments are not audited as claims.
	if got := answerSentences("Yes. OK. [1].", 8); len(got) != 0 {
		t.Errorf("fragments should not count as claims, got %v", got)
	}
}
