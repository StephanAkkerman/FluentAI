"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
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
    <header className="w-full flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
      <div className="flex items-center space-x-2">
        <Image src={logo} alt="FluentAI Logo" width={40} height={40} />
        <h1 className="text-xl font-bold text-gray-800 dark:text-gray-200">FluentAI</h1>
      </div>
      <button
        onClick={toggleDarkMode}
        className="text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100"
        aria-label="Toggle Dark Mode"
      >
        {darkMode ? "🌞 Light Mode" : "🌙 Dark Mode"}
      </button>
    </header>
  );
}

