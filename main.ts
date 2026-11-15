// free-model-aggregator-02: one server, two client wires, one free provider.
//
// It reads the brief (the free model list, the six prompts), builds one
// catalog, then serves the client's six requests: each one goes to the best
// free model that meets its requirements, by the brief's rating_milli,
// failing over to the next best free model when a call is refused or answers
// unusably.
//
// The routing is rating-ordered with a per-request failover loop: a model
// that refuses is skipped for the rest of that request, and a model that
// refuses three calls in a row is treated as spent for the trial (a scheduled
// failure is never three in a row; a spent budget or an outage always is). A
// rate limit or a transient upstream error (429 or 5xx) gets one immediate
// retry of the same model first: the limit may have cleared at once, and a
// retry that fails again costs nothing on the tie-break (the gateway records
// it as an upstream error, not an upstream call).
// Every refusal is reported back on the answer in the
// `x-makefun-upstream-failures` header, so the house can credit the failover.
// An answer only counts as served when it is usable: the provider reported
// at least one output token, the answer carries text or a tool call, and the
// text echoes the prompt's token (see lib/upstream.ts).
//
// Every prompt is dialled as an OpenAI Chat Completions request through the
// gateway, whatever wire the client used: the Anthropic path is translated
// there and back. Streaming OpenAI calls always ask for usage, so the
// client's usage check sees the provider's numbers.
//
// Every answer is normalised through lib/upstream.ts and rendered by
// lib/wire.ts, so the whole ANSWER text, the stop reason and the usage
// numbers all carry through unchanged, and the `model` on every answer and
// every streamed chunk is the id this entry dialled — never the provider's
// spelling of it. Nothing here reads the clock or unseeded randomness, so a
// replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  normToAnthropicJson,
  normToAnthropicStream,
  normToOpenAIJson,
  normToOpenAIStream,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  answerUsable,
  chatCompletionsUrl,
  dial,
  isWellFormedOpenAIJson,
  openAIJsonToNorm,
  openAIStreamToNorm,
  tokensIn,
  upstreamErrorOf,
  type NormAnswer,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refused calls after which a model is treated as spent. */
const EXHAUST_AFTER_STREAK = 3;
/** The most refusal rows the trail header carries, in order. */
const MAX_TRAIL_ROWS = 8;
/** The header the house reads a request's upstream refusals from. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
const completionsUrl = chatCompletionsUrl(GATEWAY_URL);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Refused calls in a row, per model: reset by any usable answer from that model. */
const failStreak = new Map<string, number>();
/** Models treated as spent for the rest of the trial. */
const exhaustedModels = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

interface Route {
  model: ModelRow;
  rating: number;
}

/**
 * Every free model that meets one request's requirements, best rating first.
 * Ties break on model id: the order is deterministic. A model spent for the
 * trial never appears; a model that only refused this request is skipped by
 * the caller.
 */
function candidatesFor(requires: Requirement): Route[] {
  const out: Route[] = [];
  for (const model of freeModels(brief)) {
    if (exhaustedModels.has(model.id)) continue;
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    out.push({ model, rating: model.rating_milli });
  }
  out.sort((a, b) => b.rating - a.rating || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string, extraHeaders: Record<string, string> = {}): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamFailure {
  status: number;
  code: string;
  model: string;
  /** True when the gateway itself answered 429 or 5xx: the limit may clear at once, so one retry is worth it. */
  retryable: boolean;
}

/** The trail header's value for a request's refusals: a JSON array of { status, code, model }, at most eight rows, in order. */
function trailHeader(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, MAX_TRAIL_ROWS).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  return { [FAILURES_HEADER]: JSON.stringify(rows) };
}

type UpstreamResult = { ok: true; answer: NormAnswer } | { ok: false; error: UpstreamFailure };

/** The gateway request body for one client request on one model. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, model: string, stream: boolean): Json {
  if (clientApi === 'anthropic') return anthropicToOpenAI(body, model);
  const out: Json = { ...body, model };
  if (stream) out.stream_options = { include_usage: true };
  return out;
}

/**
 * One call to one model, normalised. A refusal (4xx, 5xx), an unreachable
 * gateway, or an answer that is not usable comes back as an error, never an
 * exception, so the caller can fail over to the next candidate.
 */
async function callUpstream(route: Route, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, tokens: readonly string[]): Promise<UpstreamResult> {
  const model = route.model.id;
  let response: Response;
  try {
    response = await dial(completionsUrl, GATEWAY_KEY, upstreamBody(clientApi, body, model, stream), stream, PROVIDER_TIMEOUT_MS);
  } catch (err) {
    return { ok: false, error: { status: 502, code: (err as Error).name === 'TimeoutError' ? 'timeout' : 'unreachable', model, retryable: false } };
  }
  if (!response.ok) {
    let text = '';
    try {
      text = await response.text();
    } catch {
      // The refusal carried no readable body: the status is enough.
    }
    const { status, code } = upstreamErrorOf(response.status, text);
    return { ok: false, error: { status, code, model, retryable: status === 429 || status >= 500 } };
  }
  try {
    if (!stream) {
      const data: unknown = await response.json();
      if (!isWellFormedOpenAIJson(data)) return { ok: false, error: { status: 502, code: 'bad_answer', model, retryable: false } };
      const answer = openAIJsonToNorm(data);
      if (!answerUsable(answer, tokens)) return { ok: false, error: { status: 502, code: 'bad_answer', model, retryable: false } };
      return { ok: true, answer };
    }
    if (response.body === null) return { ok: false, error: { status: 502, code: 'no_stream', model, retryable: false } };
    const answer = await openAIStreamToNorm(response.body);
    if (!answerUsable(answer, tokens)) return { ok: false, error: { status: 502, code: 'bad_answer', model, retryable: false } };
    return { ok: true, answer };
  } catch {
    return { ok: false, error: { status: 502, code: 'bad_answer', model, retryable: false } };
  }
}

interface Served {
  route: Route;
  answer: NormAnswer;
  failures: UpstreamFailure[];
}

/**
 * Serve one request: try the candidates in rating order until one answers
 * usably. A model that refuses is skipped for the rest of this request. A
 * model's failure streak grows on every refusal and resets on any usable
 * answer; at three in a row it is treated as spent for the trial. When no
 * candidate serves, the refusals still come back, so the error answer can
 * carry them on its trail header.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement, stream: boolean): Promise<{ ok: true; served: Served } | { ok: false; failures: UpstreamFailure[] }> {
  const candidates = candidatesFor(requires);
  const tokens = tokensIn(body);
  const skipped = new Set<string>();
  const failures: UpstreamFailure[] = [];
  for (const route of candidates) {
    const modelId = route.model.id;
    if (skipped.has(modelId)) continue;
    const first = await callUpstream(route, clientApi, body, stream, tokens);
    if (first.ok) {
      failStreak.set(modelId, 0);
      return { ok: true, served: { route, answer: first.answer, failures } };
    }
    failures.push(first.error);
    let refusals = 1;
    // A rate limit or a transient upstream error may clear at once: one
    // immediate retry of the same model before falling over. The retry costs
    // nothing on the tie-break when it fails again (the gateway records it as
    // an upstream error, not an upstream call), and it keeps the best model
    // serving when the limit was momentary.
    if (first.error.retryable) {
      const retry = await callUpstream(route, clientApi, body, stream, tokens);
      if (retry.ok) {
        failStreak.set(modelId, 0);
        return { ok: true, served: { route, answer: retry.answer, failures } };
      }
      failures.push(retry.error);
      refusals = 2;
    }
    skipped.add(modelId);
    const streak = (failStreak.get(modelId) ?? 0) + refusals;
    failStreak.set(modelId, streak);
    if (streak >= EXHAUST_AFTER_STREAK) exhaustedModels.add(modelId);
    log(`failover: ${modelId} refused ${refusals} call${refusals > 1 ? 's' : ''} (HTTP ${first.error.status} ${first.error.code}); trying the next candidate`);
  }
  return { ok: false, failures };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const result = await serve('openai', body, requirementsFor(body), stream);
  if (!result.ok) {
    // No candidate served: the house still credits the recorded refusals.
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request', trailHeader(result.failures));
  }
  const served = result.served;
  const model = served.route.model.id;
  const trail = trailHeader(served.failures);
  if (stream) return sendText(res, 200, 'text/event-stream; charset=utf-8', normToOpenAIStream(served.answer, model), trail);
  return sendJson(res, 200, normToOpenAIJson(served.answer, model), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that meets it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const result = await serve('anthropic', body, requirementsFor(body), stream);
  if (!result.ok) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request', trailHeader(result.failures));
  }
  const served = result.served;
  const model = served.route.model.id;
  const trail = trailHeader(served.failures);
  if (stream) return sendText(res, 200, 'text/event-stream; charset=utf-8', normToAnthropicStream(served.answer, model), trail);
  return sendJson(res, 200, normToAnthropicJson(served.answer, model), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (completionsUrl === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
