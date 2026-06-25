import type { CollectionEntry } from "astro:content";

export type LearningPath = {
  title: string;
  slug: string;
  description: string;
  accent: string;
};

export const learningPaths: LearningPath[] = [
  {
    title: "Reinforcement Learning",
    slug: "reinforcement-learning",
    description:
      "Value functions, policy gradients, control, exploration, and decision-making systems.",
    accent: "RL",
  },
  {
    title: "Inference Engineering",
    slug: "inference-engineering",
    description:
      "Serving paths, latency, batching, memory, quantization, kernels, and deployment constraints.",
    accent: "INF",
  },
  {
    title: "Computer Vision",
    slug: "computer-vision",
    description:
      "Perception pipelines, image analysis, model evaluation, datasets, and deployment.",
    accent: "CV",
  },
  {
    title: "AI Agents",
    slug: "ai-agents",
    description:
      "Tool use, planning loops, memory, evaluation, and reliable agentic workflows.",
    accent: "AG",
  },
  {
    title: "Large Language Models",
    slug: "llms",
    description:
      "Transformers, prompting, alignment, retrieval, evaluation, and model behavior.",
    accent: "LLM",
  },
  {
    title: "Semiconductors",
    slug: "semiconductors",
    description:
      "AI accelerators, memory bandwidth, packaging, process trends, and future computing.",
    accent: "CHIP",
  },
];

export const getCategoryPosts = (
  posts: CollectionEntry<"posts">[],
  category: string
) => posts.filter(post => post.data.category === category);
