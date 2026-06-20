# Build a fully static, single-binary turbograph image.
# The web UI is embedded in the binary, so the final image is just the binary
# on a minimal base. Ollama runs separately (see docker-compose.yml).
#
# Cross-compilation is driven by buildx: the build stage always runs on the
# native BUILDPLATFORM and emits a binary for the requested TARGET arch, so
# multi-arch builds never pay for QEMU emulation of the Go compiler.
FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS build
WORKDIR /src
# Cache modules first.
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=docker
ARG TARGETOS
ARG TARGETARCH
# CGO is off: build a fully static binary with the embedded UI for the target.
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /turbograph ./cmd/turbograph

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /turbograph /turbograph
# Buckets persist here; mount a volume to keep them across restarts.
WORKDIR /data
# Runs as the distroless "nonroot" user (uid 65532) by default.
EXPOSE 8080
ENTRYPOINT ["/turbograph"]
# Point at an Ollama reachable from the container; override as needed.
CMD ["serve", "--addr", ":8080", "--data", "/data", "--ollama-url", "http://host.docker.internal:11434"]
