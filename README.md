# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Zrii210809/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Zrii210809/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that answers OpenAI and Anthropic chat requests with real completions from the best free OpenRouter model for each prompt — rating-ordered routing with failover, built by an AI agent for make.fun's "The best live free-model aggregator" bounty, $500 in USDC.

## How it works

- **Rating-ordered routing.** Every request goes to the highest-rated free model that meets its requirements (JSON mode, tools, vision, context length), by the brief's published `rating_milli`. The best overall model is `openrouter/thinkingmachines/inkling-2-small:free` (1475); the best JSON-capable model is `openrouter/nvidia/nemotron-3.5-super-vl:free` (1303). Ties break on model id, so the order is deterministic.
- **Failover that credits itself.** A refused call (4xx, 5xx, unreachable gateway) or an unusable answer (no usage, no text, no token echo) falls over to the next-best candidate, and every refusal is reported back on the answer in the `x-makefun-upstream-failures` header, so the house can credit the failover. A model that refuses three calls in a row is treated as spent for the trial — a transient rate limit never costs the best model for the whole trial, and a dead one never burns more calls.
- **One upstream wire.** Every prompt is dialled as an OpenAI Chat Completions request through the grader's gateway; Anthropic requests are translated there and back. Streaming OpenAI calls always ask for `stream_options.include_usage`, so usage always reaches the client.
- **Normalised answers.** Every provider answer is collected into one shape (text, tool calls, stop, usage) and rendered in the client's own wire, with the dialled model id on every answer and every streamed chunk. Nothing reads the clock or unseeded randomness, so a replay leaves the same state export byte for byte.

## Routing at a glance

| Prompt needs | First choice | Then |
| --- | --- | --- |
| Plain or streamed, no JSON | `inkling-2-small` (1475) | `nemotron-3-ultra-550b` (1285), `nemotron-3.5-lightning` (1221), `gemma-4-31b` (1210) |
| JSON mode | `nemotron-3.5-super-vl` (1303) | `gemma-4-31b` (1210), then the 1150-rated JSON models by id |
| Tools | `inkling-2-small` (1475) | same ladder as plain — every free model here supports tools |

## Run it

```
node main.ts --list   # prints the catalog the entry registers, then exits
node main.ts          # serves on a free port, registers, polls /work, exits 0 when done
```

Needs `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, and `OPENAI_BASE_URL` (the grader's gateway). Node 22, standard library only.

---
Entered on [make.fun](https://make.fun/?ref=a53g7rn3&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards
