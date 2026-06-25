---
title: "Attention Is All You Need: Reading Notes"
description: "A structured research note on the transformer paper and its engineering implications."
pubDatetime: 2026-06-25T11:00:00+05:30
tags:
  - Research
  - LLMs
category: "Research"
featured: true
draft: false
---

## Paper Summary

The transformer replaces recurrence with attention, enabling parallel sequence processing and a scalable architecture for language modeling.

## Core Intuition

Tokens can exchange information directly through attention rather than passing state step by step through a recurrent chain.

## Method / Architecture

The architecture stacks multi-head self-attention, feed-forward networks, residual connections, and normalization.

## Mathematics

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

## Implementation Notes

The paper's simplicity hides important implementation details: masking, numerical stability, memory layout, and batching.

## Experiments

Initial experiments focused on machine translation, but the design became a foundation for large-scale language modeling.

## Observations

The architecture is as much a systems idea as a modeling idea because it parallelizes well on modern accelerators.

## Limitations

Attention cost grows with sequence length, which motivates sparse attention, linear attention, recurrence hybrids, and better cache design.

## References

- Vaswani et al., "Attention Is All You Need", 2017.
