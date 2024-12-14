import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'export',
  basePath: "/FluentAI",
  images: {
    unoptimized: true, // Required for GitHub Pages
  },
};

export default nextConfig;
