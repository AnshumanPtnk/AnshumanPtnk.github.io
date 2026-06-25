---
title: "KV Cache and Token Latency"
description: "Why key-value cache design matters for LLM inference latency and memory."
pubDatetime: 2026-06-25T10:10:00+05:30
tags:
  - Inference Engineering
  - LLMs
category: "Inference Engineering"
featured: true
draft: false
---

Autoregressive inference is shaped by a simple constraint: each generated token depends on the tokens before it. The key-value cache avoids recomputing attention state for the full context at every step, but it moves pressure onto memory capacity and bandwidth.

Useful questions for inference design:

- How large is the cache per request?
- Can multiple requests be batched without wasting memory?
- What is the prefill versus decode latency profile?
- When does quantization help or hurt?

```mermaid
flowchart LR
  P[Prompt tokens] --> F[Prefill]
  F --> K[KV cache]
  K --> D[Decode loop]
  D --> O[Output tokens]
  D --> K
```
