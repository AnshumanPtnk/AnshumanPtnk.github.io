---
title: "Memory Bandwidth Is the AI Bottleneck"
description: "A short note on why AI performance is often constrained by moving data, not just compute."
pubDatetime: 2026-06-25T10:50:00+05:30
tags:
  - Semiconductors
  - AI Hardware
category: "Semiconductors"
featured: true
draft: false
---

AI accelerators are often discussed in terms of peak FLOPS, but real workloads spend much of their time moving data. Memory bandwidth, cache locality, interconnects, and packaging decisions shape useful throughput.

For inference, the bottleneck can shift depending on model size, batch size, quantization, sequence length, and cache behavior. The hardware story is inseparable from the software serving path.
