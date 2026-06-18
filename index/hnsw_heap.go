package index

import "sync"

// minHeap orders candidates by ascending distance: the closest is at the root.
// It drives the frontier of an HNSW layer search.
type minHeap []candidate

func (h minHeap) Len() int { return len(h) }

func (h *minHeap) push(c candidate) {
	*h = append(*h, c)
	i := len(*h) - 1
	a := *h
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist <= a[i].dist {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
}

func (h *minHeap) pop() candidate {
	a := *h
	top := a[0]
	n := len(a) - 1
	a[0] = a[n]
	a = a[:n]
	*h = a
	i := 0
	for {
		l, r := 2*i+1, 2*i+2
		small := i
		if l < n && a[l].dist < a[small].dist {
			small = l
		}
		if r < n && a[r].dist < a[small].dist {
			small = r
		}
		if small == i {
			break
		}
		a[i], a[small] = a[small], a[i]
		i = small
	}
	return top
}

// maxHeap orders candidates by descending distance: the furthest is at the root,
// so the worst current result can be evicted in O(1) when the result set is full.
type maxHeap []candidate

func (h maxHeap) Len() int { return len(h) }

func (h *maxHeap) push(c candidate) {
	*h = append(*h, c)
	i := len(*h) - 1
	a := *h
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist >= a[i].dist {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
}

func (h *maxHeap) pop() candidate {
	a := *h
	top := a[0]
	n := len(a) - 1
	a[0] = a[n]
	a = a[:n]
	*h = a
	i := 0
	for {
		l, r := 2*i+1, 2*i+2
		big := i
		if l < n && a[l].dist > a[big].dist {
			big = l
		}
		if r < n && a[r].dist > a[big].dist {
			big = r
		}
		if big == i {
			break
		}
		a[i], a[big] = a[big], a[i]
		i = big
	}
	return top
}

// visitedSet is an epoch-based membership set. Bumping the epoch clears it in O(1)
// instead of reallocating, which matters because a search visits the set
// thousands of times and many searches run concurrently.
type visitedSet struct {
	marks []uint32
	epoch uint32
}

func (v *visitedSet) reset(n int) {
	if cap(v.marks) < n {
		v.marks = make([]uint32, n)
		v.epoch = 0
	}
	v.marks = v.marks[:n]
	v.epoch++
	if v.epoch == 0 {
		// Wrapped: clear and restart so stale marks cannot alias.
		for i := range v.marks {
			v.marks[i] = 0
		}
		v.epoch = 1
	}
}

func (v *visitedSet) mark(i int)      { v.marks[i] = v.epoch }
func (v *visitedSet) seen(i int) bool { return v.marks[i] == v.epoch }

var visitedPool = sync.Pool{New: func() any { return &visitedSet{} }}

func (h *HNSW) acquireVisited() *visitedSet {
	v := visitedPool.Get().(*visitedSet)
	v.reset(len(h.nodes))
	return v
}

func (h *HNSW) releaseVisited(v *visitedSet) { visitedPool.Put(v) }
