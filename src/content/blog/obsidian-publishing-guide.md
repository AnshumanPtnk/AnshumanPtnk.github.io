---
draft: true
pubDatetime: 2026-06-25T10:00:00+05:30
title: "Obsidian Publishing Guide"
tags:
  - Meta
description: "Private guide for publishing Markdown notes from Obsidian into this Astro blog."
---

Use `src/content/blog/` as the Obsidian vault folder, or symlink an Obsidian folder here. Astro still publishes the notes under `/posts/...` routes, so existing AstroPaper reading, RSS, search, tags, and article pages continue to work.

Recommended topic folders:

- `reinforcement-learning/`
- `inference-engineering/`
- `computer-vision/`
- `ai-agents/`
- `llms/`
- `semiconductors/`
- `research/`
- `thoughts/`

Research notes should use this outline:

- Paper Summary
- Core Intuition
- Method / Architecture
- Mathematics
- Implementation Notes
- Experiments
- Observations
- Limitations
- References
- `distributed-systems/`

Each public note needs frontmatter like this:

```yaml
---
pubDatetime: 2026-06-25T10:00:00+05:30
title: "Your Post Title"
tags:
  - Reinforcement Learning
category: "Reinforcement Learning"
description: "A short summary for previews, RSS, and SEO."
---
```

Set `draft: true` while a note is still private. Remove it or set `draft: false` when you want the post to publish.

Markdown, MDX, images, code blocks, LaTeX equations, and Mermaid fences are supported.
