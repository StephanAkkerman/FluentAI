import "./globals.css";
import Header from "@/components/Header";
import StatusChecker from "@/components/StatusChecker";
import { ToastProvider } from "@/contexts/ToastContext";
import type { Metadata } from "next";

const isGithubPages = process.env.NODE_ENV === "production" && process.env.GITHUB_PAGES === "true";

export const metadata: Metadata = {
  title: "FluentAI",
  description: "Learning languages in a flash.",
  icons: {
    icon: `${isGithubPages ? '/FluentAI' : ''}/logo.png`,
  }
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 text-gray-800 dark:text-gray-200 font-sans min-h-screen flex flex-col">
        <ToastProvider>
          <Header />

          <main >
            {children}
          </main>
          <StatusChecker />
        </ToastProvider>
      </body>
    </html>
  );
}
