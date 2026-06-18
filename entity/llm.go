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

const extractSystem = "You extract a knowledge graph from text. " +
	"Identify the salient entities (people, organizations, places, products, concepts) " +
	"and the relationships between them. " +
	"Output one item per line and nothing else, in exactly these formats:\n" +
	"entity|name|type|one short description\n" +
	"relation|source entity|target entity|how they relate\n" +
	"Use the same entity names in relations as in the entity lines. Do not add commentary."

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
