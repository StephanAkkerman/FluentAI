import type { Metadata } from "next";

const isGithubPages = process.env.NODE_ENV === "production" && process.env.GITHUB_PAGES === "true";

export const metadata: Metadata = {
  title: "FluentAI",
  description: "Learning languages in a flash.",
  icons: {
    icon: `${isGithubPages ? '/FluentAI' : ''}/logo.png`,
  }
};
