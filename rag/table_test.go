package rag

import (
	"strings"
	"testing"
)

// TestMarkdownTableAtomic checks that a Markdown table is kept whole (never split
// row by row), carries its heading breadcrumb, and is prefixed with its caption.
func TestMarkdownTableAtomic(t *testing.T) {
	text := "# Report\n\n## Results\n\nQuarterly revenue by region:\n\n" +
		"| Region | Q1 | Q2 | Q3 | Q4 |\n" +
		"| --- | --- | --- | --- | --- |\n" +
		"| North | 10 | 12 | 14 | 16 |\n" +
		"| South | 8 | 9 | 11 | 13 |\n" +
		"| East | 5 | 6 | 7 | 8 |\n\n" +
		"That concludes the results section with some trailing prose here."
	// Small target so a naive splitter would shred the table.
	ch := markdownChunker{target: 8, overlap: 0}
	pieces := ch.Split(text)

	var tablePiece *Piece
	for i := range pieces {
		if strings.Contains(pieces[i].Text, "| North |") {
			tablePiece = &pieces[i]
		}
	}
	if tablePiece == nil {
		t.Fatal("no piece contains the table")
	}
	// The whole table must be in ONE piece: all rows present together.
	for _, row := range []string{"| North |", "| South |", "| East |", "| Region |"} {
		if !strings.Contains(tablePiece.Text, row) {
			t.Fatalf("table was split; piece missing %q:\n%s", row, tablePiece.Text)
		}
	}
	// Caption prepended.
	if !strings.Contains(tablePiece.Text, "Quarterly revenue by region") {
		t.Errorf("table piece missing its caption:\n%s", tablePiece.Text)
	}
	// Heading breadcrumb attached.
	if len(tablePiece.Headings) == 0 || tablePiece.Headings[len(tablePiece.Headings)-1] != "Results" {
		t.Errorf("table piece missing the Results breadcrumb: %v", tablePiece.Headings)
	}
	// The trailing prose is a separate piece, not glued into the table.
	if strings.Contains(tablePiece.Text, "trailing prose") {
		t.Errorf("prose after the table leaked into the table piece")
	}
}
