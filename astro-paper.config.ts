import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://anshumanptnk.github.io/",
    title: "Anshuman Patnaik",
    description:
      "A public engineering notebook documenting the journey from software architecture to AI research. Explore deep dives into reinforcement learning, inference engineering, AI agents, world models, GPU systems, and the mathematics behind modern machine intelligence—written for engineers, researchers, and builders who value understanding over memorization.",
    author: "Anshuman Patnaik",
    profile: "https://github.com/AnshumanPtnk",
    googleAnalyticsId: "G-ZVPM15YE9L",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Kolkata",
    dir: "ltr",
  },
  posts: {
    perPage: 4,
    perIndex: 5,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    dynamicOgImage: true,
    showArchives: false,
    showBackButton: true,
    editPost: {
      enabled: false,
    },
    search: "pagefind",
  },
  socials: [
    { name: "github",   url: "https://github.com/AnshumanPtnk" },
    { name: "linkedin", url: "https://www.linkedin.com/in/anshuman-patnaik-a1a3a256/" },
    { name: "mail",     url: "mailto:anshuman.ptnk@gmail.com" },
  ],
  shareLinks: [
    { name: "whatsapp", url: "https://wa.me/?text=" },
    { name: "facebook", url: "https://www.facebook.com/sharer.php?u=" },
    { name: "x",        url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "pinterest", url: "https://pinterest.com/pin/create/button/?url=" },
    { name: "mail",     url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
