# turbograph developer tasks. Everything here uses only the Go toolchain.
BIN := bin/turbograph
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: build install test test-race test-short cover fuzz vet fmt lint bench clean docker run

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

lint: fmt vet ## fmt + vet (CI gate)
	@test -z "$$(gofmt -l .)" || (echo "unformatted files:"; gofmt -l .; exit 1)

bench: build ## run the TurboQuant codec benchmark
	$(BIN) quant bench

clean:
	rm -rf bin

docker: ## build the container image
	docker build -t turbograph:latest .

run: build ## build and serve on :8080
	$(BIN) serve

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'
