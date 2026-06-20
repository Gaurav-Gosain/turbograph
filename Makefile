# turbograph developer tasks. Everything here uses only the Go toolchain
# (plus golangci-lint for the lint target and the client toolchains).
BIN := bin/turbograph
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: build install test test-race test-short cover fuzz vet fmt fmt-check \
	lint bench clean docker run clients-test ci help

build: ## build the single binary (embedded UI)
	go build -trimpath -ldflags "$(LDFLAGS)" -o $(BIN) ./cmd/turbograph

install: ## go install the binary
	go install ./cmd/turbograph

test: ## run the full test suite
	go test ./...

test-short: ## skip the slow recall/QPS sweeps
	go test -short ./...

test-race: ## run tests under the race detector
	go test -short -race ./...

cover: ## print per-package coverage
	go test -short -cover ./...

fuzz: ## run the codec fuzzer for 30s
	go test ./quant/ -run x -fuzz FuzzEncodeDecode -fuzztime 30s

vet: ## go vet
	go vet ./...

fmt: ## format all Go code
	gofmt -w .

fmt-check: ## fail if any Go file is unformatted
	@test -z "$$(gofmt -l .)" || (echo "unformatted files:"; gofmt -l .; exit 1)

lint: ## run golangci-lint (matches CI)
	golangci-lint run

bench: ## run the benchmark suite once (smoke)
	go test -run=^$$ -bench=. -benchtime=1x ./...

clean:
	rm -rf bin

docker: ## build the container image
	docker build -t turbograph:latest .

run: build ## build and serve on :8080
	$(BIN) serve

clients-test: ## run the Python and TypeScript client test suites
	cd clients/python && python -m pip install -e ".[dev]" && python -m pytest -q
	cd clients/typescript && npm ci && npx tsc --noEmit && node --test "test/**/*.test.mjs"

ci: fmt-check vet lint test-race bench ## run the full CI gate locally

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'
