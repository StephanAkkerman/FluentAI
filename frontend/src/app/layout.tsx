import Header from "@/components/Header";
import StatusChecker from "@/components/StatusChecker";
import { ToastProvider } from "@/contexts/ToastContext";
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "mnemorai",
  description: "Learning languages in a flash.",
  icons: {
    icon: `/logo.png`,
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
          <main className="flex-grow max-w-6xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-12 space-y-8 transition-all duration-300">
            {children}
          </main>
          <StatusChecker />
          <footer className="text-center p-6 text-gray-500 dark:text-gray-400 text-sm">
            <p>© 2025 FluentAI • Made with ❤️ for language learners</p>
          </footer>
        </ToastProvider>
      </body>
    </html>
  );
}
