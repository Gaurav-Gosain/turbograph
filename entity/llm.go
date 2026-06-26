package entity

import (
	"context"
	"strings"
)

// Generator produces a completion for a system and user prompt. The Ollama client
// satisfies this once a model is bound; keeping it an interface lets the extractor
// run against any model without depending on a specific client.
type Generator interface {
	Generate(ctx context.Context, system, prompt string) (string, error)
}

// LLMExtractor extracts entities and relationships with a language model. The
// prompt asks for a simple line-delimited format that small local models follow
// reliably, and the parser is lenient about surrounding noise.
type LLMExtractor struct {
	Gen Generator
}

// NewLLMExtractor returns an extractor backed by gen.
func NewLLMExtractor(gen Generator) *LLMExtractor { return &LLMExtractor{Gen: gen} }

const extractRules = "Rules:\n" +
	"- Use one canonical name per entity. If it is mentioned by a pronoun, alias, or short form, " +
	"use its most complete identifier everywhere (he, Dr. Babbage -> Charles Babbage).\n" +
	"- Use a basic type from: person, organization, location, date, event, product, concept. " +
	"Avoid overly specific types (use person, not scientist) and vague ones (entity, thing).\n" +
	"- Name relations with a short lowercase verb phrase (invented, born_in, works_at). " +
	"Avoid vague relations like is, has, related_to.\n" +
	"- Each relation description is one concrete fact naming both entities. " +
	"Good: wrote the first algorithm for the Analytical Engine. Bad: describes a relationship.\n" +
	"- Use only the text. Do not add outside knowledge or invent relationships.\n" +
	"- Use the same entity names in relations as in the entity lines. Do not add commentary."

const extractSystem = "You extract a knowledge graph from text. " +
	"Identify the salient entities (people, organizations, places, products, concepts) " +
	"and the relationships between them. " +
	"Output one item per line and nothing else, in exactly these formats:\n" +
	"entity|name|type|one short description\n" +
	"relation|source entity|target entity|how they relate\n" +
	extractRules

const extractExample = "Example output:\n" +
	"entity|Ada Lovelace|person|nineteenth century mathematician\n" +
	"entity|Analytical Engine|product|early mechanical computer\n" +
	"relation|Ada Lovelace|Analytical Engine|wrote the first algorithm for it\n"

// Extract runs the model and parses its output.
func (e *LLMExtractor) Extract(ctx context.Context, text string) (Extraction, error) {
	prompt := extractExample + "\nText:\n" + text + "\n\nOutput:"
	out, err := e.Gen.Generate(ctx, extractSystem, prompt)
	if err != nil {
		return Extraction{}, err
	}
	return Parse(out), nil
}

// BatchExtractor extracts from several passages in one model call, returning one
// Extraction per input in order. Batching amortizes the per-call overhead and
// dramatically cuts the number of LLM round trips over a large corpus.
type BatchExtractor interface {
	ExtractBatch(ctx context.Context, texts []string) ([]Extraction, error)
}

const extractBatchSystem = "You extract a knowledge graph from several numbered passages. " +
	"For each passage, identify the salient entities (people, organizations, places, products, concepts) " +
	"and the relationships between them. " +
	"Output one item per line and nothing else, each tagged with its passage number, in exactly these formats:\n" +
	"entity|<passage number>|name|type|one short description\n" +
	"relation|<passage number>|source entity|target entity|how they relate\n" +
	extractRules

const extractBatchExample = "Example output for two passages:\n" +
	"entity|1|Ada Lovelace|person|nineteenth century mathematician\n" +
	"relation|1|Ada Lovelace|Analytical Engine|wrote the first algorithm for it\n" +
	"entity|2|Bell Labs|organization|research laboratory\n"

// ExtractBatch extracts from texts in a single model call, one Extraction per
// input. A single-element batch falls back to Extract, and any passage the model
// omits comes back empty rather than failing the whole batch.
func (e *LLMExtractor) ExtractBatch(ctx context.Context, texts []string) ([]Extraction, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if len(texts) == 1 {
		ex, err := e.Extract(ctx, texts[0])
		return []Extraction{ex}, err
	}
	var sb strings.Builder
	sb.WriteString(extractBatchExample)
	sb.WriteString("\nPassages:\n")
	for i, t := range texts {
		sb.WriteString("[")
		sb.WriteString(itoa(i + 1))
		sb.WriteString("] ")
		sb.WriteString(t)
		sb.WriteString("\n")
	}
	sb.WriteString("\nOutput:")
	out, err := e.Gen.Generate(ctx, extractBatchSystem, sb.String())
	if err != nil {
		return nil, err
	}
	return ParseBatch(out, len(texts)), nil
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// ParseBatch turns tagged line-delimited output into n Extractions, one per
// passage. Lines whose passage tag is missing or out of range are ignored, so
// stray output never corrupts a neighbour's extraction.
func ParseBatch(out string, n int) []Extraction {
	res := make([]Extraction, n)
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		line = strings.TrimPrefix(line, "- ")
		line = strings.Trim(line, "`")
		if line == "" {
			continue
		}
		parts := strings.Split(line, "|")
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		if len(parts) < 3 {
			continue
		}
		idx := atoi(parts[1])
		if idx < 1 || idx > n {
			continue
		}
		p := &res[idx-1]
		switch strings.ToLower(parts[0]) {
		case "entity":
			if parts[2] == "" {
				continue
			}
			typ, desc := "", ""
			if len(parts) >= 4 {
				typ = parts[3]
			}
			if len(parts) >= 5 {
				desc = strings.Join(parts[4:], " ")
			}
			p.Entities = append(p.Entities, ExtractedEntity{Name: parts[2], Type: typ, Description: desc})
		case "relation", "relationship":
			if len(parts) >= 4 && parts[2] != "" && parts[3] != "" {
				desc := ""
				if len(parts) >= 5 {
					desc = strings.Join(parts[4:], " ")
				}
				p.Relations = append(p.Relations, ExtractedRelation{Source: parts[2], Target: parts[3], Description: desc})
			}
		}
	}
	return res
}

// atoi parses a small non-negative integer, returning -1 on any non-digit input.
func atoi(s string) int {
	if s == "" {
		return -1
	}
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return -1
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// Parse turns line-delimited model output into an Extraction. It ignores lines
// that do not match the expected shape, so stray prose or code fences do not
// break it.
func Parse(out string) Extraction {
	var ex Extraction
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		line = strings.TrimPrefix(line, "- ")
		line = strings.Trim(line, "`")
		if line == "" {
			continue
		}
		parts := strings.Split(line, "|")
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		switch strings.ToLower(parts[0]) {
		case "entity":
			if len(parts) >= 2 && parts[1] != "" {
				typ, desc := "", ""
				if len(parts) >= 3 {
					typ = parts[2]
				}
				if len(parts) >= 4 {
					desc = strings.Join(parts[3:], " ")
				}
				ex.Entities = append(ex.Entities, ExtractedEntity{Name: parts[1], Type: typ, Description: desc})
			}
		case "relation", "relationship":
			if len(parts) >= 3 && parts[1] != "" && parts[2] != "" {
				desc := ""
				if len(parts) >= 4 {
					desc = strings.Join(parts[3:], " ")
				}
				ex.Relations = append(ex.Relations, ExtractedRelation{Source: parts[1], Target: parts[2], Description: desc})
			}
		}
	}
	return ex
}
