import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'export',
  basePath: "",
  images: {
    unoptimized: true, // Required for GitHub Pages
  },
};

export default nextConfig;
