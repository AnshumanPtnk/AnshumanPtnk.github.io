---
title: "Bellman Backup: An Intuitive Introduction"
description: "Understanding the Bellman backup operation in reinforcement learning."
pubDatetime: 2026-06-25T10:00:00+05:30
tags:
  - Reinforcement Learning
  - Bellman Equation
category: "Reinforcement Learning"
featured: true
draft: false
---

The Bellman backup is the small recursive move at the center of many reinforcement learning algorithms. It updates a value estimate by asking: if this is the current state, what reward do I receive now, and how valuable is the best future that follows?

For a state-value function, the idea is often written as:

$$
V(s) \leftarrow \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V(s') \right]
$$

This note is a starting point for unpacking why that recurrence is powerful, where it becomes brittle, and how it shows up in dynamic programming, temporal-difference learning, and approximate control.

```python
def bellman_backup(transitions, gamma, values):
    return max(
        sum(prob * (reward + gamma * values[next_state])
            for prob, next_state, reward in action_outcomes)
        for action_outcomes in transitions
    )
```
