// The gateway dial and the one upstream wire, normalised to one answer shape.
//
// Every prompt goes upstream as an OpenAI Chat Completions request to the
// grader's gateway (`$OPENAI_BASE_URL`, an OpenAI-compatible endpoint that
// reaches OpenRouter on the key the entrant pushed). Whatever the provider
// answers becomes the same thing:
//
//   - JSON: { text, toolCalls, finish, usage }
//   - SSE: the same, collected from the provider's event stream.
//
// The entry renders that shape back to the client in the client's own wire
// (see lib/wire.ts), so a request can be served by any free model and still
// pass the trial's checks: the whole ANSWER text, the stop reason family and
// the usage numbers, all carried through unchanged.
//
// A call only counts as served when its normalised answer is usable: the
// provider reported at least one output token, the answer carries text or a
// tool call, and the text carries the prompt's token. Anything else — a
// refusal, a broken body, an answer without usage, an answer that never echoes
// the token — is a failure the caller can fail over from, never an answer the
// client sees. The token check reads the token off the request itself, so it
// needs no knowledge of the prompt list.
//
// Everything here is a pure function of its arguments plus the provider's
// answer: no clock, no randomness, so a replay sees the same bytes.

import { isRecord, type Json } from './providers.ts';

/** One tool call in a normalised answer: arguments as a JSON string. */
export interface NormToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** A provider's answer, whatever its shape: text, tool calls, stop, usage. */
export interface NormAnswer {
  text: string;
  toolCalls: NormToolCall[];
  finish: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  usage: { input: number; output: number } | null;
}

export const EMPTY_ANSWER: NormAnswer = { text: '', toolCalls: [], finish: 'stop', usage: null };

/** The shape of a prompt's token: three to twelve capitals, a dash, four digits. */
const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/g;

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as a JSON string, from an object or a string. */
function argsString(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** OpenAI's finish_reason as the normalised finish. */
function finishOfOpenAI(finish: unknown): NormAnswer['finish'] {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

// ---------------------------------------------------------------------------
// The dial: one Chat Completions call through the gateway.
// ---------------------------------------------------------------------------

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`…/v1` on the trial network); a base that does not
 * already end in `/v1` gets one, so both spellings reach the route.
 */
export function chatCompletionsUrl(base: string): string {
  const root = base.replace(/\/+$/, '');
  if (root === '') return '';
  return root.endsWith('/v1') ? `${root}/chat/completions` : `${root}/v1/chat/completions`;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
export async function dial(url: string, key: string, body: Json, stream: boolean, timeoutMs: number): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${key}`,
  };
  if (stream) headers.accept = 'text/event-stream';
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

/** A refusal's { status, code } from its OpenAI-shaped error body: `code`, else `type`, else ''. */
export function upstreamErrorOf(status: number, bodyText: string): { status: number; code: string } {
  let code = '';
  try {
    const value: unknown = JSON.parse(bodyText);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') code = value.error.code;
      else if (typeof value.error.type === 'string') code = value.error.type;
    }
  } catch {
    // The refusal carried no JSON body: the status is enough.
  }
  return { status, code: code.slice(0, 64) };
}

// ---------------------------------------------------------------------------
// Response parsing: the provider's answer into the normalised shape.
// ---------------------------------------------------------------------------

/** Whether a Chat Completions JSON body is well-formed enough to serve: an assistant message and a finish reason. */
export function isWellFormedOpenAIJson(body: unknown): body is Json {
  if (!isRecord(body)) return false;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  if (choices.length === 0 || !isRecord(choices[0])) return false;
  const choice = choices[0];
  if (!isRecord(choice.message) || choice.message.role !== 'assistant') return false;
  return typeof choice.finish_reason === 'string';
}

/** An OpenAI Chat Completions JSON answer, normalised. */
export function openAIJsonToNorm(body: Json): NormAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: NormToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({ id: stringOf(call.id) !== '' ? stringOf(call.id) : 'call_unknown', name: stringOf(call.function.name), arguments: argsString(call.function.arguments) });
  }
  const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) } : null;
  return { text, toolCalls, finish: finishOfOpenAI(choice.finish_reason), usage };
}

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a byte stream's text into SSE frames (blank-line separated). */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const emit = function* (chunk: string): Generator<SseFrame> {
    buffer += chunk.replace(/\r\n/g, '\n');
    let at = buffer.indexOf('\n\n');
    while (at !== -1) {
      const frame = buffer.slice(0, at);
      buffer = buffer.slice(at + 2);
      let event: string | null = null;
      const dataLines: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
      at = buffer.indexOf('\n\n');
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield* emit(decoder.decode(value, { stream: true }));
    }
    yield* emit(decoder.decode());
  } finally {
    reader.releaseLock();
  }
}

function parseJsonFrame(data: string): Json | null {
  if (data === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI SSE stream, collected into the normalised answer. */
export async function openAIStreamToNorm(stream: ReadableStream<Uint8Array>): Promise<NormAnswer> {
  const textParts: string[] = [];
  const toolByIndex = new Map<number, NormToolCall>();
  let finish: NormAnswer['finish'] = 'stop';
  let sawFinish = false;
  let usage: NormAnswer['usage'] = null;
  for await (const frame of sseFrames(stream)) {
    const chunk = parseJsonFrame(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') textParts.push(delta.content);
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : toolByIndex.size;
      let entry = toolByIndex.get(index);
      if (entry === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        entry = { id: stringOf(call.id) !== '' ? stringOf(call.id) : `call_${index}`, name: stringOf(fn.name), arguments: '' };
        toolByIndex.set(index, entry);
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '' && entry.name === '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      finish = finishOfOpenAI(choice.finish_reason);
      sawFinish = true;
    }
  }
  // A stream with no finish chunk never stopped: the client would fail its
  // stream-order check, so the answer is not usable even if text arrived.
  if (!sawFinish) return { text: '', toolCalls: [], finish: 'stop', usage: null };
  return { text: textParts.join(''), toolCalls: [...toolByIndex.values()], finish, usage };
}

// ---------------------------------------------------------------------------
// Usability: whether a normalised answer may serve the request that asked.
// ---------------------------------------------------------------------------

/** Every token-shaped string in a request body: the messages, the system prompt, the stop sequences. */
export function tokensIn(body: Json): string[] {
  const found = new Set<string>();
  const scan = (value: unknown): void => {
    if (typeof value === 'string') {
      for (const match of value.matchAll(TOKEN_PATTERN)) found.add(match[0]);
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) scan(item);
      return;
    }
    if (isRecord(value)) {
      for (const item of Object.values(value)) scan(item);
    }
  };
  scan(body.messages);
  scan(body.system);
  scan(body.stop);
  return [...found];
}

/**
 * Whether a normalised answer may serve its request: the provider reported at
 * least one output token (the client's usage check needs it), the answer
 * carries text or a tool call, and — when the request named a token — the
 * answer echoes one of them (the client's content check needs it).
 */
export function answerUsable(answer: NormAnswer, tokens: readonly string[]): boolean {
  if (answer.usage === null || answer.usage.output < 1) return false;
  if (answer.text === '' && answer.toolCalls.length === 0) return false;
  if (tokens.length > 0 && !tokens.some((token) => answer.text.includes(token))) return false;
  return true;
}
