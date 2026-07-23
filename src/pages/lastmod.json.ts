import { getCollection } from "astro:content";
import { getSortedPosts } from "@/utils/getSortedPosts";
import { getPostUrl } from "@/utils/getPostPaths";
import config from "@/config";

export async function GET() {
  const posts = await getCollection("posts");
  const sortedPosts = getSortedPosts(posts);
  const siteUrl = config.site.url.replace(/\/$/, "");

  const lastmod: Record<string, string> = {};
  for (const { id, data, filePath } of sortedPosts) {
    const url = `${siteUrl}${getPostUrl(id, filePath, config.site.lang)}`;
    lastmod[url.replace(/\/$/, "")] = new Date(
      data.modDatetime ?? data.pubDatetime
    ).toISOString();
  }

  return new Response(JSON.stringify(lastmod), {
    headers: { "Content-Type": "application/json" },
  });
}
