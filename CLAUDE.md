# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal AI-engineering notebook/blog by Anshuman Patnaik, built on the [AstroPaper](https://github.com/satnaing/astro-paper) Astro theme (v6, forked/customized). Deployed to GitHub Pages at `anshumanptnk.github.io`.

## Commands

Package manager is **npm** (`package-lock.json` is authoritative; a `pnpm-lock.yaml`/`pnpm-workspace.yaml` also exist but CI uses npm — use npm for all work here).

```bash
npm install         # install dependencies
npm run dev          # dev server at localhost:4321
npm run build        # astro check + astro build + pagefind indexing (copies index to public/pagefind/)
npm run preview      # preview the production build locally
npm run sync         # regenerate Astro content-collection types (.astro/types.d.ts)
npm run lint         # eslint .
npm run format       # prettier --write .
npm run format:check # prettier --check .
```

There is no test suite. CI (`.github/workflows/ci.yml`) runs `npm ci && npm run build` on PRs — this is the effective correctness gate, since `astro build` runs a type check first. Deployment (`.github/workflows/astro.yml`) builds and publishes `dist/` to GitHub Pages on push to `main`.

To check a single page/route while iterating, just run `npm run dev` and visit it — there's no per-file test runner to target.

## Architecture

### Two content trees — only one is live

- `src/content/blog/` is the **actual, live** post content, organized into category subdirectories (`ai-agents/`, `computer-vision/`, `inference-engineering/`, `llms/`, `reinforcement-learning/`, `research/`, `semiconductors/`, `thoughts/`).
- `src/content/posts/` is leftover **upstream AstroPaper template documentation/example content** (theme config docs, release notes, example posts). It is *not* wired into any content collection and is not rendered on the site. Don't confuse the two when looking for "where posts live."

This split exists because `content.config.ts` defines the `posts` *collection* with `BLOG_PATH = "src/content/blog"` — i.e., the collection named `posts` (used everywhere via `getCollection("posts")`) actually loads from the `blog/` directory, not the `content/posts/` directory. Files/dirs prefixed with `_` (e.g. `_color-schemes/`, `_releases/`) are excluded by the loader glob (`[^_]*`) and used as example/reference material.

### Post frontmatter conventions

Required: `title`, `description`, `pubDatetime`. Common: `tags` (array), `category` (one of the six learning-path titles below — drives routing on Research/Thoughts/Learning Paths pages), `featured`, `draft`. See schema in [src/content.config.ts](src/content.config.ts).

### Content routing / site sections

Beyond the standard AstroPaper pages (`/posts`, `/tags`, `/archives` if enabled, `/search`), this fork adds three custom section pages built on `category`:

- [src/pages/research.astro](src/pages/research.astro) — posts with `category: "Research"`.
- [src/pages/thoughts.astro](src/pages/thoughts.astro) — posts with `category: "Thoughts"`.
- [src/pages/learning-paths.astro](src/pages/learning-paths.astro) — groups posts by the six categories defined in [src/data/portal.ts](src/data/portal.ts) (`learningPaths` array: Reinforcement Learning, Inference Engineering, Computer Vision, AI Agents, LLMs, Semiconductors). Adding a new learning path means adding an entry there *and* using that exact `category` string in post frontmatter.

Post URLs are derived from filesystem path (not front-matter slug): `getPostSlug`/`getPostUrl` in [src/utils/getPostPaths.ts](src/utils/getPostPaths.ts) strip `BLOG_PATH`, drop `_`-prefixed segments, and join the remaining directory segments with the filename to form the route — so a post's subdirectory under `src/content/blog/` becomes part of its URL.

### Configuration layering

- [astro-paper.config.ts](astro-paper.config.ts) — the user-facing config (site metadata, socials, feature flags). Edit this for site-level changes.
- [src/config.ts](src/config.ts) — resolves `astro-paper.config.ts` against defaults into `ResolvedAstroPaperConfig`; imported everywhere else as `config`. Don't hand-edit defaults here unless changing the fallback behavior itself.
- [src/types/config.ts](src/types/config.ts) — the config type contracts and `defineAstroPaperConfig` helper.
- [astro.config.ts](astro.config.ts) — Astro/Vite/markdown pipeline config (remark/rehype plugins, Shiki themes, MDX, sitemap, i18n, fonts, env schema).

### Markdown/MDX pipeline

Math via `remark-math` + `rehype-katex`, callouts via `rehype-callouts`, TOC via `remark-toc` + `remark-collapse`, code blocks via Shiki with custom transformers in [src/utils/transformers/fileName.js](src/utils/transformers/fileName.js) plus diff/highlight notation transformers. Mermaid diagrams are used in post content (see fenced ` ```mermaid ` blocks) — check how existing posts render these before assuming built-in support beyond what's configured.

### i18n

Single locale (`en`) is configured in `astro.config.ts`, but the routing/URL-building machinery (`getRelativeLocaleUrl`, `src/i18n/`) is generic multi-locale infrastructure inherited from the theme. `Astro.currentLocale` is used as the locale source across pages.

### Dynamic OG images

`features.dynamicOgImage` in `astro-paper.config.ts` controls whether `src/pages/og.png.ts` and `src/pages/posts/[...slug]/index.png.ts` generate per-post OG images via Satori + Sharp. If disabled, a static `public/{site.ogImage}` file is required instead.

### Search

Pagefind (`features.search: "pagefind"`) indexes the built site during `npm run build` and copies the index into `public/pagefind/` — this is a post-build step, not something available in `npm run dev`.
