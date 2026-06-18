package rag

import "strings"

// GraphNode is a chunk as seen by a visualization: its identity, the community it
// belongs to, its degree in the similarity graph, and a short text preview.
type GraphNode struct {
	Index     int    `json:"index"`
	ID        string `json:"id"`
	DocID     string `json:"doc_id"`
	Community int    `json:"community"`
	Degree    int    `json:"degree"`
	Snippet   string `json:"snippet"`
}

// GraphEdge is an undirected similarity edge between two chunk indices.
type GraphEdge struct {
	Source int     `json:"source"`
	Target int     `json:"target"`
	Weight float32 `json:"weight"`
}

// GraphView is a serializable snapshot of the similarity graph for rendering.
type GraphView struct {
	Nodes []GraphNode `json:"nodes"`
	Edges []GraphEdge `json:"edges"`
}

// GraphView exports the current similarity graph. Each undirected edge is emitted
// once (source < target). It is safe for concurrent use.
func (s *Store) GraphView() GraphView {
	s.mu.RLock()
	defer s.mu.RUnlock()

	view := GraphView{Nodes: make([]GraphNode, len(s.chunks))}
	for i, c := range s.chunks {
		community := -1
		if s.comm != nil {
			community = s.comm.Label(i)
		}
		degree := 0
		if s.g != nil {
			degree = s.g.Degree(i)
		}
		view.Nodes[i] = GraphNode{
			Index:     i,
			ID:        c.ID,
			DocID:     c.DocID,
			Community: community,
			Degree:    degree,
			Snippet:   snippet(c.Text, 160),
		}
	}
	if s.g != nil {
		for i := range s.chunks {
			s.g.Neighbors(i, func(j int, w float32) {
				if j > i {
					view.Edges = append(view.Edges, GraphEdge{Source: i, Target: j, Weight: w})
				}
			})
		}
	}
	return view
}

func snippet(text string, n int) string {
	text = strings.Join(strings.Fields(text), " ")
	if len(text) <= n {
		return text
	}
	// Trim to the last word boundary within the budget for a cleaner preview.
	cut := text[:n]
	if sp := strings.LastIndexByte(cut, ' '); sp > n/2 {
		cut = cut[:sp]
	}
	return cut + "..."
}
