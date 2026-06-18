package index

// topHeap keeps the best `cap` hits seen so far using a bounded binary heap whose
// root is the worst currently-kept item. When full, a new hit replaces the root
// only if it is better, giving O(log cap) per push and O(cap) memory regardless
// of how many items are scanned.
type topHeap struct {
	data   []hit
	cap    int
	higher bool // true when larger scores are better
}

func newTopHeap(capacity int, higherIsBetter bool) *topHeap {
	return &topHeap{data: make([]hit, 0, capacity), cap: capacity, higher: higherIsBetter}
}

// worse reports whether a ranks below b (a is closer to being evicted).
func (h *topHeap) worse(a, b hit) bool {
	if h.higher {
		return a.score < b.score
	}
	return a.score > b.score
}

func (h *topHeap) push(x hit) {
	if len(h.data) < h.cap {
		h.data = append(h.data, x)
		h.up(len(h.data) - 1)
		return
	}
	// Heap is full; root is the worst kept item. Replace it only if x is better.
	if h.worse(h.data[0], x) {
		h.data[0] = x
		h.down(0)
	}
}

func (h *topHeap) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !h.worse(h.data[i], h.data[parent]) {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *topHeap) down(i int) {
	n := len(h.data)
	for {
		l, r := 2*i+1, 2*i+2
		worst := i
		if l < n && h.worse(h.data[l], h.data[worst]) {
			worst = l
		}
		if r < n && h.worse(h.data[r], h.data[worst]) {
			worst = r
		}
		if worst == i {
			break
		}
		h.data[i], h.data[worst] = h.data[worst], h.data[i]
		i = worst
	}
}

// drain returns the kept hits (unsorted) and empties the heap.
func (h *topHeap) drain() []hit {
	out := h.data
	h.data = nil
	return out
}
