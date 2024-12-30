import type { NextConfig } from "next";

const isGithubPages = process.env.NODE_ENV === "production" && process.env.GITHUB_PAGES === "true";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'export',
  basePath: isGithubPages ? "/FluentAI" : "",
  images: {
    unoptimized: true, // Required for GitHub Pages
  },
};

export default nextConfig;
