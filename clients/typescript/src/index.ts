/**
 * @turbograph/client
 *
 * A dependency-free TypeScript client for turbograph, a graph-RAG server.
 * It targets both modern browsers and Node 18+ using the global `fetch` and
 * `ReadableStream`. There are no runtime dependencies.
 *
 * Quick start:
 *
 *   import { Turbograph } from "@turbograph/client";
 *   const tg = new Turbograph({ baseUrl: "http://localhost:8080" });
 *   await tg.ingestText([{ id: "doc1", text: "hello world" }]);
 *   const hits = await tg.query("hello");
 *   for await (const ev of tg.chat("what is this corpus about?")) {
 *     if (ev.type === "token") process.stdout.write(ev.text);
 *   }
 */

// ---------------------------------------------------------------------------
// Types that mirror the server's JSON shapes exactly.
// ---------------------------------------------------------------------------

/** Arbitrary, JSON-serializable document metadata. */
export type Meta = Record<string, unknown>;

/** A document to ingest as plain text (POST /ingest). */
export interface Document {
  id: string;
  text: string;
  /** Optional arbitrary metadata stored with the document. */
  meta?: Meta;
  /** "image" marks an image-derived document; usually set by the server. */
  kind?: string;
  /** Asset id of the source image for an image-derived document. */
  image_ref?: string;
}

/** A binary file to ingest (POST /api/ingest/files). */
export interface IngestFile {
  /** Document id for the extracted text. */
  id: string;
  /** The file bytes, base64 (standard encoding). */
  b64: string;
  /** Optional arbitrary metadata stored with the document. */
  meta?: Meta;
}

/** Per-file failure reported by ingestFiles. */
export interface IngestFailure {
  id: string;
  error: string;
}

/** Result of POST /ingest. */
export interface IngestResult {
  chunks: number;
  saved: boolean;
  save_error: string;
}

/** Result of POST /api/ingest/files. */
export interface IngestFilesResult {
  chunks: number;
  indexed: number;
  failed: IngestFailure[] | null;
  saved: boolean;
  save_error: string;
}

/** Result of POST /api/ingest/image. */
export interface IngestImageResult {
  id: string;
  image_ref: string;
  caption: string;
}

/** Retrieval tuning parameters shared by query and chat. */
export interface RetrieveParams {
  /** Number of chunks to return. Server default is 6 for chat. */
  top_k?: number;
  /** Strength of the personalized-PageRank graph signal. 0 disables it. */
  graph_mix?: number;
  /** MMR relevance/diversity tradeoff. 0 disables diversification. */
  mmr_lambda?: number;
  /** Weight of the entity-graph signal in [0,1]. 0 ignores it. */
  entity_mix?: number;
}

/** One retrieved chunk (the server's queryResult). */
export interface QueryResult {
  id: string;
  doc_id: string;
  score: number;
  similarity: number;
  text: string;
  /** Rune offset of this chunk in its document. */
  start: number;
  end: number;
  /** The source document's metadata, when present. */
  meta?: Meta;
  /** "image" for an image-derived chunk. */
  kind?: string;
  /** Asset id of the source image. */
  image_ref?: string;
}

/** Span of one chunk inside its document (rag.ChunkSpan). */
export interface ChunkSpan {
  id: string;
  pos: number;
  /** Rune offset, -1 if the chunk could not be located. */
  start: number;
  /** Rune offset (exclusive). */
  end: number;
}

/** A document with its full text, metadata, and chunk spans (rag.DocView). */
export interface DocView {
  id: string;
  text: string;
  meta?: Meta;
  spans: ChunkSpan[];
}

/** Summary of one ingested document (rag.DocInfo). */
export interface DocInfo {
  id: string;
  chunks: number;
  bytes: number;
}

/** One entry in a document's content history (rag.DocVersion). */
export interface DocVersion {
  /** 1-based version number, oldest is 1. */
  n: number;
  /** Short hex content hash. */
  hash: string;
  /** Unix seconds when recorded. */
  time: number;
  /** Document size at this version. */
  bytes: number;
  /** Chunks this version produced. */
  chunks: number;
  /** Whether this is the live version. */
  current: boolean;
}

/** A generated thematic community summary (rag.CommunitySummary). */
export interface CommunitySummary {
  label: number;
  /** Number of chunks in the community. */
  size: number;
  /** The generated thematic summary. */
  summary: string;
  /** Member chunk ids. */
  chunks: string[];
  /** Distinct source documents. */
  doc_ids: string[];
}

/** Bucket listing entry. */
export interface BucketInfo {
  name: string;
  chunks: number;
  documents: number;
  communities: number;
}

/** Response of GET /api/models. */
export interface ModelsResult {
  models: string[];
  default: string;
  pdf: boolean;
  embed_model?: string;
  embed_ready?: boolean;
}

/** Response of GET /api/status. */
export interface StatusResult {
  version: string;
  storage: { backend: string; location: string; endpoint: string };
  generation: { api: string; model: string; reachable: boolean };
  embedding: { api: string; model: string; dim: number };
  buckets: number;
  bucket: string;
  stats?: {
    chunks: number;
    documents: number;
    entities: number;
    communities?: number;
    chunk_strategy?: string;
  };
}

/** A conversation turn passed to chat for query rewriting. */
export interface ChatTurn {
  role: string;
  content: string;
}

// ---------------------------------------------------------------------------
// Chat streaming event types. These are the parsed SSE events from /api/chat.
// ---------------------------------------------------------------------------

/** Emitted first (in local and global modes) with the retrieved sources. */
export interface SourcesEvent {
  type: "sources";
  sources: QueryResult[];
}
/** Emitted for each generated text token. */
export interface TokenEvent {
  type: "token";
  text: string;
}
/** Emitted when the evidence gate fires and no answer is generated. */
export interface AbstainEvent {
  type: "abstain";
  message: string;
}
/** Emitted on a server-side error during streaming. */
export interface StreamErrorEvent {
  type: "error";
  error: string;
}
/** Emitted once when the stream is complete. */
export interface DoneEvent {
  type: "done";
  done: boolean;
}

export type ChatEvent =
  | SourcesEvent
  | TokenEvent
  | AbstainEvent
  | StreamErrorEvent
  | DoneEvent;

/** Progress event from the build-entities SSE stream. */
export interface EntityProgressEvent {
  type: "progress";
  done: number;
  total: number;
  entities: number;
  relations: number;
}
/** Terminal event from the build-entities SSE stream. */
export interface EntityDoneEvent {
  type: "done";
  entities: number;
}
export type EntityBuildEvent =
  | EntityProgressEvent
  | StreamErrorEvent
  | EntityDoneEvent;

/** Progress event from the build-communities SSE stream. */
export interface CommunityProgressEvent {
  type: "progress";
  done: number;
  total: number;
}
/** Terminal event from the build-communities SSE stream. */
export interface CommunityDoneEvent {
  type: "done";
  communities: number;
}
export type CommunityBuildEvent =
  | CommunityProgressEvent
  | StreamErrorEvent
  | CommunityDoneEvent;

/** Options for chat(): retrieval params plus generation controls. */
export interface ChatOptions extends RetrieveParams {
  /** Abstain if the top hit's cosine is below this. */
  min_sim?: number;
  /** Pointwise LLM reranking of candidates. */
  rerank?: boolean;
  /** Recent turns, used for conversational query rewriting. */
  history?: ChatTurn[];
  /** Generation model; defaults to the server's configured model. */
  model?: string;
  /** Document metadata keys to include in each passage. */
  metaKeys?: string[];
  /** Answer from community summaries (corpus-wide questions). */
  global?: boolean;
  /** Optional AbortSignal to cancel the stream. */
  signal?: AbortSignal;
}

/** Resolved result of chatText(). */
export interface ChatText {
  answer: string;
  sources: QueryResult[];
  /** Set when the server abstained instead of answering. */
  abstained?: string;
}

/** Options for the image ingestion call. */
export interface IngestImageOptions {
  /** Document id for the image. */
  id: string;
  /** The image bytes as raw bytes or a base64 string. */
  image: Uint8Array | string;
  /** File extension (png, jpg, ...). */
  ext: string;
  /** Vision model to caption with. */
  model: string;
  /** Optional captioning instruction. */
  prompt?: string;
  /** Optional document metadata. */
  meta?: Meta;
}

/** Construction options for the client. */
export interface TurbographOptions {
  /** Base URL of the server. Default "http://localhost:8080". */
  baseUrl?: string;
  /** Bucket (isolated corpus) to operate on. Default "default". */
  bucket?: string;
  /** Optional fetch implementation override (mainly for testing). */
  fetch?: typeof fetch;
}

// ---------------------------------------------------------------------------
// Errors.
// ---------------------------------------------------------------------------

/** Thrown on any non-2xx response, carrying the server's {error} message. */
export class TurbographError extends Error {
  /** HTTP status code of the failing response. */
  readonly status: number;
  /** The full path that was requested. */
  readonly url: string;
  /** The decoded JSON body, when available. */
  readonly body?: unknown;

  constructor(status: number, url: string, message: string, body?: unknown) {
    super(message);
    this.name = "TurbographError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

// ---------------------------------------------------------------------------
// Base64 helpers that work in both Node and the browser without dependencies.
// ---------------------------------------------------------------------------

/** Encode raw bytes to a standard base64 string. */
export function toBase64(bytes: Uint8Array): string {
  // Browsers and most runtimes expose btoa; Node 18+ also has it globally.
  if (typeof btoa === "function") {
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode(
        ...bytes.subarray(i, i + chunk),
      );
    }
    return btoa(binary);
  }
  // Node fallback.
  const B = (globalThis as Record<string, unknown>).Buffer as
    | { from(d: Uint8Array): { toString(enc: string): string } }
    | undefined;
  if (B) return B.from(bytes).toString("base64");
  throw new Error("no base64 encoder available in this runtime");
}

/** Decode a standard base64 string to raw bytes. */
export function fromBase64(b64: string): Uint8Array {
  if (typeof atob === "function") {
    const binary = atob(b64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  }
  const B = (globalThis as Record<string, unknown>).Buffer as
    | { from(d: string, enc: string): Uint8Array }
    | undefined;
  if (B) return new Uint8Array(B.from(b64, "base64"));
  throw new Error("no base64 decoder available in this runtime");
}

// ---------------------------------------------------------------------------
// SSE parsing over a fetch ReadableStream. Works in Node 18+ and browsers.
// ---------------------------------------------------------------------------

/** One parsed server-sent event. */
export interface SSEMessage {
  /** The event name, or "message" if the stream omitted "event:". */
  event: string;
  /** The concatenated data lines (the raw payload, usually JSON). */
  data: string;
}

/**
 * parseSSE consumes a ReadableStream of bytes and yields one SSEMessage per
 * complete event. Events are separated by a blank line; "event:" and "data:"
 * fields are accumulated until that separator. Comment lines (":") and unknown
 * fields are ignored. This is intentionally small and standalone so it can be
 * unit tested without a live server.
 */
export async function* parseSSE(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEMessage> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "message";
  let data: string[] = [];

  const flush = (): SSEMessage | null => {
    if (data.length === 0 && event === "message") return null;
    const msg: SSEMessage = { event, data: data.join("\n") };
    event = "message";
    data = [];
    return msg;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Process complete lines; normalize CRLF and CR to LF.
      let nl: number;
      // Find a line terminator. We scan for "\n" after replacing "\r\n"/"\r".
      buffer = buffer.replace(/\r\n?/g, "\n");
      while ((nl = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);

        if (line === "") {
          const msg = flush();
          if (msg) yield msg;
          continue;
        }
        if (line.startsWith(":")) continue; // comment / keep-alive
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        let val = colon === -1 ? "" : line.slice(colon + 1);
        if (val.startsWith(" ")) val = val.slice(1); // strip one leading space
        if (field === "event") event = val;
        else if (field === "data") data.push(val);
        // id and retry fields are ignored by this client.
      }
    }
    // Flush any trailing event the stream ended without a blank line.
    const tail = flush();
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// The client.
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL = "http://localhost:8080";
const DEFAULT_BUCKET = "default";

export class Turbograph {
  readonly baseUrl: string;
  readonly bucket: string;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: TurbographOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.bucket = opts.bucket ?? DEFAULT_BUCKET;
    const f = opts.fetch ?? (globalThis.fetch as typeof fetch | undefined);
    if (!f) {
      throw new Error(
        "global fetch is not available; pass opts.fetch or use Node 18+ / a modern browser",
      );
    }
    this.fetchImpl = f;
  }

  /** Returns a new client scoped to a different bucket, sharing settings. */
  withBucket(bucket: string): Turbograph {
    return new Turbograph({
      baseUrl: this.baseUrl,
      bucket,
      fetch: this.fetchImpl,
    });
  }

  // --- URL building -------------------------------------------------------

  /** Build a full URL for a path, always appending the bucket query param. */
  private url(path: string, query?: Record<string, string | undefined>): string {
    const u = new URL(this.baseUrl + path);
    u.searchParams.set("bucket", this.bucket);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined) u.searchParams.set(k, v);
      }
    }
    return u.toString();
  }

  // --- HTTP core ----------------------------------------------------------

  private async request(
    method: string,
    url: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<Response> {
    const init: RequestInit = { method, signal };
    if (body !== undefined) {
      init.body = JSON.stringify(body);
      init.headers = { "Content-Type": "application/json" };
    }
    const res = await this.fetchImpl(url, init);
    if (!res.ok) {
      await this.raise(res, url);
    }
    return res;
  }

  /** Read an error response and throw a TurbographError from its {error}. */
  private async raise(res: Response, url: string): Promise<never> {
    let message = `request failed with status ${res.status}`;
    let body: unknown;
    try {
      const text = await res.text();
      if (text) {
        try {
          body = JSON.parse(text);
          const m = (body as { error?: unknown }).error;
          if (typeof m === "string" && m) message = m;
        } catch {
          message = text;
        }
      }
    } catch {
      // Ignore body read failures; keep the status-based message.
    }
    throw new TurbographError(res.status, url, message, body);
  }

  private async json<T>(
    method: string,
    url: string,
    body?: unknown,
  ): Promise<T> {
    const res = await this.request(method, url, body);
    return (await res.json()) as T;
  }

  /** Open an SSE stream and yield parsed events, throwing on non-2xx. */
  private async *sse(
    method: string,
    url: string,
    body?: unknown,
    signal?: AbortSignal,
  ): AsyncGenerator<SSEMessage> {
    const res = await this.request(method, url, body, signal);
    if (!res.body) {
      throw new TurbographError(
        res.status,
        url,
        "response had no body to stream",
      );
    }
    yield* parseSSE(res.body);
  }

  // --- Ingestion ----------------------------------------------------------

  /**
   * Ingest plain-text documents (POST /ingest). When replace is true the
   * corpus is rebuilt from scratch; otherwise documents are added incrementally.
   */
  ingestText(
    documents: Document[],
    opts: { replace?: boolean } = {},
  ): Promise<IngestResult> {
    return this.json<IngestResult>("POST", this.url("/ingest"), {
      documents,
      replace: opts.replace ?? false,
    });
  }

  /**
   * Ingest binary files for text extraction (POST /api/ingest/files). Each file
   * carries its base64 bytes; files that fail to parse are reported in `failed`.
   */
  ingestFiles(files: IngestFile[]): Promise<IngestFilesResult> {
    return this.json<IngestFilesResult>(
      "POST",
      this.url("/api/ingest/files"),
      { files },
    );
  }

  /**
   * Ingest an image (POST /api/ingest/image): it is stored, captioned by a
   * vision model, and indexed as an image chunk referencing the stored asset.
   */
  ingestImage(opts: IngestImageOptions): Promise<IngestImageResult> {
    const b64 =
      typeof opts.image === "string" ? opts.image : toBase64(opts.image);
    return this.json<IngestImageResult>(
      "POST",
      this.url("/api/ingest/image"),
      {
        id: opts.id,
        b64,
        ext: opts.ext,
        model: opts.model,
        prompt: opts.prompt ?? "",
        meta: opts.meta,
      },
    );
  }

  // --- Retrieval ----------------------------------------------------------

  /** Retrieve the most relevant chunks for a query (POST /query). */
  async query(text: string, params: RetrieveParams = {}): Promise<QueryResult[]> {
    const res = await this.json<{ results: QueryResult[] | null }>(
      "POST",
      this.url("/query"),
      {
        query: text,
        top_k: params.top_k ?? 0,
        graph_mix: params.graph_mix ?? 0,
        mmr_lambda: params.mmr_lambda ?? 0,
        entity_mix: params.entity_mix ?? 0,
      },
    );
    return res.results ?? [];
  }

  // --- Chat (streaming) ---------------------------------------------------

  /**
   * Stream a retrieval-augmented chat answer (POST /api/chat) as typed events.
   * Yields a "sources" event, then "token" events, then "done"; or "abstain"
   * when the evidence gate fires, or "error" on a server error.
   *
   *   for await (const ev of tg.chat("question")) {
   *     if (ev.type === "token") process.stdout.write(ev.text);
   *   }
   */
  async *chat(query: string, opts: ChatOptions = {}): AsyncGenerator<ChatEvent> {
    const body = {
      query,
      top_k: opts.top_k ?? 0,
      graph_mix: opts.graph_mix ?? 0,
      mmr_lambda: opts.mmr_lambda ?? 0,
      entity_mix: opts.entity_mix ?? 0,
      min_sim: opts.min_sim ?? 0,
      rerank: opts.rerank ?? false,
      history: opts.history ?? [],
      model: opts.model ?? "",
      meta_keys: opts.metaKeys ?? [],
      global: opts.global ?? false,
    };
    for await (const msg of this.sse(
      "POST",
      this.url("/api/chat"),
      body,
      opts.signal,
    )) {
      const ev = parseChatEvent(msg);
      if (ev) yield ev;
    }
  }

  /**
   * Convenience wrapper that consumes the chat stream and resolves to the full
   * answer and its sources. Throws a TurbographError on a streamed "error".
   */
  async chatText(query: string, opts: ChatOptions = {}): Promise<ChatText> {
    let answer = "";
    let sources: QueryResult[] = [];
    let abstained: string | undefined;
    for await (const ev of this.chat(query, opts)) {
      switch (ev.type) {
        case "sources":
          sources = ev.sources;
          break;
        case "token":
          answer += ev.text;
          break;
        case "abstain":
          abstained = ev.message;
          break;
        case "error":
          throw new TurbographError(
            200,
            this.url("/api/chat"),
            ev.error,
            ev,
          );
        case "done":
          break;
      }
    }
    return { answer, sources, abstained };
  }

  // --- Documents ----------------------------------------------------------

  /** List the documents in the current bucket (GET /api/documents). */
  async documents(): Promise<DocInfo[]> {
    const res = await this.json<{ documents: DocInfo[] | null }>(
      "GET",
      this.url("/api/documents"),
    );
    return res.documents ?? [];
  }

  /** Fetch one document's text, metadata, and chunk spans (GET /api/document). */
  document(doc: string): Promise<DocView> {
    return this.json<DocView>("GET", this.url("/api/document", { doc }));
  }

  /** Delete a document and its chunks (DELETE /api/document). */
  deleteDocument(doc: string): Promise<{ deleted: string; chunks: number }> {
    return this.json("DELETE", this.url("/api/document", { doc }));
  }

  // --- Versions -----------------------------------------------------------

  /** List a document's content history, oldest first (GET /api/versions). */
  async versions(doc: string): Promise<DocVersion[]> {
    const res = await this.json<{ doc: string; versions: DocVersion[] | null }>(
      "GET",
      this.url("/api/versions", { doc }),
    );
    return res.versions ?? [];
  }

  /** Fetch the stored text of one version (GET /api/version). */
  async versionText(doc: string, n: number): Promise<string> {
    const res = await this.json<{ doc: string; n: number; text: string }>(
      "GET",
      this.url("/api/version", { doc, n: String(n) }),
    );
    return res.text;
  }

  /** Restore an earlier version as the live document (POST /api/restore). */
  restore(
    doc: string,
    n: number,
  ): Promise<{ doc: string; restored: number; versions: DocVersion[] }> {
    return this.json(
      "POST",
      this.url("/api/restore", { doc, n: String(n) }),
    );
  }

  // --- Entity graph -------------------------------------------------------

  /**
   * Build the entity-relationship graph with the language model
   * (POST /api/build-entities), yielding SSE progress events.
   */
  async *buildEntities(
    model?: string,
    opts: { batch?: number; signal?: AbortSignal } = {},
  ): AsyncGenerator<EntityBuildEvent> {
    const query: Record<string, string | undefined> = {
      model: model || undefined,
      batch: opts.batch !== undefined ? String(opts.batch) : undefined,
    };
    for await (const msg of this.sse(
      "POST",
      this.url("/api/build-entities", query),
      undefined,
      opts.signal,
    )) {
      const ev = parseBuildEvent(msg) as EntityBuildEvent | null;
      if (ev) yield ev;
    }
  }

  /**
   * Build entities and block until done, returning the terminal event.
   * Throws a TurbographError if the stream reported an error.
   */
  async buildEntitiesSync(
    model?: string,
    opts: { batch?: number; signal?: AbortSignal } = {},
  ): Promise<EntityDoneEvent> {
    let last: EntityDoneEvent | undefined;
    for await (const ev of this.buildEntities(model, opts)) {
      if (ev.type === "error") {
        throw new TurbographError(
          200,
          this.url("/api/build-entities"),
          ev.error,
          ev,
        );
      }
      if (ev.type === "done") last = ev;
    }
    return last ?? { type: "done", entities: 0 };
  }

  // --- Communities --------------------------------------------------------

  /**
   * Build community summaries with the language model
   * (POST /api/build-communities), yielding SSE progress events.
   */
  async *buildCommunities(
    model?: string,
    opts: { maxPassages?: number; signal?: AbortSignal } = {},
  ): AsyncGenerator<CommunityBuildEvent> {
    const query: Record<string, string | undefined> = {
      model: model || undefined,
      max_passages:
        opts.maxPassages !== undefined ? String(opts.maxPassages) : undefined,
    };
    for await (const msg of this.sse(
      "POST",
      this.url("/api/build-communities", query),
      undefined,
      opts.signal,
    )) {
      const ev = parseBuildEvent(msg) as CommunityBuildEvent | null;
      if (ev) yield ev;
    }
  }

  /**
   * Build communities and block until done, returning the terminal event.
   * Throws a TurbographError if the stream reported an error.
   */
  async buildCommunitiesSync(
    model?: string,
    opts: { maxPassages?: number; signal?: AbortSignal } = {},
  ): Promise<CommunityDoneEvent> {
    let last: CommunityDoneEvent | undefined;
    for await (const ev of this.buildCommunities(model, opts)) {
      if (ev.type === "error") {
        throw new TurbographError(
          200,
          this.url("/api/build-communities"),
          ev.error,
          ev,
        );
      }
      if (ev.type === "done") last = ev;
    }
    return last ?? { type: "done", communities: 0 };
  }

  /** List the generated community summaries (GET /api/communities). */
  async communities(): Promise<CommunitySummary[]> {
    const res = await this.json<{ communities: CommunitySummary[] | null }>(
      "GET",
      this.url("/api/communities"),
    );
    return res.communities ?? [];
  }

  // --- Models, status, save ----------------------------------------------

  /** List available generation models and defaults (GET /api/models). */
  models(): Promise<ModelsResult> {
    return this.json<ModelsResult>("GET", this.url("/api/models"));
  }

  /** Aggregate server status for the current bucket (GET /api/status). */
  status(): Promise<StatusResult> {
    return this.json<StatusResult>("GET", this.url("/api/status"));
  }

  /** Persist the current bucket to disk (POST /api/save). */
  save(): Promise<{ saved: boolean; bucket: string; path: string }> {
    return this.json("POST", this.url("/api/save"));
  }

  // --- Buckets ------------------------------------------------------------

  /** List buckets with basic stats (GET /api/buckets). */
  async buckets(): Promise<BucketInfo[]> {
    const res = await this.json<{ buckets: BucketInfo[] | null; default: string }>(
      "GET",
      this.url("/api/buckets"),
    );
    return res.buckets ?? [];
  }

  /** Create the named bucket, or the client's bucket if none is given. */
  createBucket(name?: string): Promise<{ created: string }> {
    const u = new URL(this.baseUrl + "/api/buckets");
    u.searchParams.set("bucket", name ?? this.bucket);
    return this.json("POST", u.toString());
  }

  /** Delete the named bucket (the default bucket cannot be deleted). */
  deleteBucket(name: string): Promise<{ deleted: string }> {
    const u = new URL(this.baseUrl + "/api/buckets");
    u.searchParams.set("bucket", name);
    return this.json("DELETE", u.toString());
  }

  // --- Assets -------------------------------------------------------------

  /** Build the public URL for a stored image asset (GET /api/asset/<id>). */
  assetUrl(imageRef: string): string {
    return this.baseUrl + "/api/asset/" + encodeURIComponent(imageRef);
  }

  /**
   * Fetch a stored image asset. Returns a Blob in environments that support it
   * (browsers), otherwise a Uint8Array (Node).
   */
  async getAsset(imageRef: string): Promise<Blob | Uint8Array> {
    const url = this.assetUrl(imageRef);
    const res = await this.fetchImpl(url, { method: "GET" });
    if (!res.ok) await this.raise(res, url);
    if (typeof Blob !== "undefined") {
      return await res.blob();
    }
    return new Uint8Array(await res.arrayBuffer());
  }
}

// ---------------------------------------------------------------------------
// SSE message -> typed event mapping.
// ---------------------------------------------------------------------------

function decode(data: string): unknown {
  if (!data) return {};
  try {
    return JSON.parse(data);
  } catch {
    return { raw: data };
  }
}

/** Map a raw SSE message from /api/chat into a typed ChatEvent. */
export function parseChatEvent(msg: SSEMessage): ChatEvent | null {
  const d = decode(msg.data) as Record<string, unknown>;
  switch (msg.event) {
    case "sources":
      return { type: "sources", sources: (d.sources as QueryResult[]) ?? [] };
    case "token":
      return { type: "token", text: String(d.text ?? "") };
    case "abstain":
      return { type: "abstain", message: String(d.message ?? "") };
    case "error":
      return { type: "error", error: String(d.error ?? "") };
    case "done":
      return { type: "done", done: Boolean(d.done) };
    default:
      return null;
  }
}

/** Map a raw SSE message from a build stream into a typed event. */
function parseBuildEvent(
  msg: SSEMessage,
): EntityBuildEvent | CommunityBuildEvent | null {
  const d = decode(msg.data) as Record<string, number | string>;
  switch (msg.event) {
    case "progress":
      // Entity progress carries entities/relations; community progress does not.
      if ("entities" in d || "relations" in d) {
        return {
          type: "progress",
          done: Number(d.done ?? 0),
          total: Number(d.total ?? 0),
          entities: Number(d.entities ?? 0),
          relations: Number(d.relations ?? 0),
        };
      }
      return {
        type: "progress",
        done: Number(d.done ?? 0),
        total: Number(d.total ?? 0),
      };
    case "error":
      return { type: "error", error: String(d.error ?? "") };
    case "done":
      if ("communities" in d) {
        return { type: "done", communities: Number(d.communities ?? 0) };
      }
      return { type: "done", entities: Number(d.entities ?? 0) };
    default:
      return null;
  }
}
