"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import logo from "../../public/logo.png";

export default function Header() {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  return (
    <header className="w-full flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-sm">
      <div className="flex items-center space-x-4">
        <Image
          src={logo}
          alt="FluentAI Logo"
          width={50}
          height={50}
          className="transition-transform duration-300 hover:rotate-12"
        />
        <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-teal-400">
          FluentAI
        </h1>
      </div>
      <div className="flex items-center space-x-6">
        <Link href="/" className="text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100">
          Home
        </Link>
        <Link href="/library" className="text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100">
          Library
        </Link>
      </div>
      <button
        onClick={toggleDarkMode}
        className="px-4 py-2 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-300 flex items-center space-x-2"
        aria-label="Toggle Dark Mode"
      >
        {darkMode ? "🌞 Light Mode" : "🌙 Dark Mode"}
      </button>
    </header>
  );
}

