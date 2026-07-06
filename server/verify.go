package server

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// claimVerdict is one audited sentence of an answer and whether the retrieved
// evidence supports it. Verdict is "supported", "partial", or "unsupported".
type claimVerdict struct {
	Claim   string `json:"claim"`
	Verdict string `json:"verdict"`
}

const verifySystem = "You audit whether each numbered claim is supported by the numbered evidence. " +
	"For every claim reply on its own line as 'N: SUPPORTED', 'N: PARTIAL', or 'N: UNSUPPORTED', " +
	"judging only whether the evidence entails the claim, not whether it is true in general. " +
	"Output only those lines, one per claim, nothing else."

const maxVerifyClaims = 8

var verdictLine = regexp.MustCompile(`(?i)\b(\d+)\s*[:.)\-]\s*(supported|partial|unsupported)`)

// auditFaithfulness splits the answer into sentences and, in a single model call,
// checks each against the retrieved passages, returning a support verdict per
// claim. It is read-only: it never edits the answer. It returns nil when there is
// nothing to check (no evidence, or no substantive claims) so the caller can skip
// emitting an audit.
func (s *Server) auditFaithfulness(ctx context.Context, model, answer string, res []rag.Retrieved) []claimVerdict {
	if s.gen == nil || model == "" || len(res) == 0 {
		return nil
	}
	claims := answerSentences(answer, maxVerifyClaims)
	if len(claims) == 0 {
		return nil
	}
	var sb strings.Builder
	sb.WriteString("Evidence:\n")
	for i, r := range res {
		fmt.Fprintf(&sb, "[%d] %s\n", i+1, oneLine(r.Chunk.Text))
	}
	sb.WriteString("\nClaims:\n")
	for i, c := range claims {
		fmt.Fprintf(&sb, "%d. %s\n", i+1, c)
	}
	sb.WriteString("\nVerdicts:")
	out, err := s.gen.Generate(ctx, model, verifySystem, sb.String())
	if err != nil {
		return nil
	}
	// Parse "N: VERDICT" lines; default any claim the model skipped to "partial" so
	// a missing verdict is never reported as a confident pass or fail.
	verdicts := make([]string, len(claims))
	for _, m := range verdictLine.FindAllStringSubmatch(out, -1) {
		var n int
		if _, err := fmt.Sscanf(m[1], "%d", &n); err != nil || n < 1 || n > len(claims) {
			continue
		}
		verdicts[n-1] = strings.ToLower(m[2])
	}
	result := make([]claimVerdict, len(claims))
	for i, c := range claims {
		v := verdicts[i]
		if v == "" {
			v = "partial"
		}
		result[i] = claimVerdict{Claim: c, Verdict: v}
	}
	return result
}

// answerSentences splits generated text into up to max substantive sentences,
// dropping trivial fragments so the audit spends its budget on real claims.
func answerSentences(text string, max int) []string {
	var out []string
	for _, s := range splitAnswerSentences(text) {
		s = strings.TrimSpace(s)
		if len(strings.Fields(s)) < 4 { // skip fragments like "Yes." or "[1]"
			continue
		}
		out = append(out, s)
		if len(out) >= max {
			break
		}
	}
	return out
}

// splitAnswerSentences is a light sentence splitter for answer text: it breaks on
// terminal punctuation followed by whitespace. It does not need the chunker's
// abbreviation guards; a slightly over-split claim is still fine to audit.
func splitAnswerSentences(text string) []string {
	var out []string
	start := 0
	r := []rune(text)
	for i := 0; i < len(r); i++ {
		if (r[i] == '.' || r[i] == '!' || r[i] == '?') && (i+1 >= len(r) || r[i+1] == ' ' || r[i+1] == '\n') {
			if seg := strings.TrimSpace(string(r[start : i+1])); seg != "" {
				out = append(out, seg)
			}
			start = i + 1
		}
	}
	if seg := strings.TrimSpace(string(r[start:])); seg != "" {
		out = append(out, seg)
	}
	return out
}

// oneLine collapses whitespace so a passage sits on a single evidence line.
func oneLine(s string) string { return strings.Join(strings.Fields(s), " ") }
