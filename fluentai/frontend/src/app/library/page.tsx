"use client";

import FlashcardLibrary from '@/components/FlashcardLibrary';

export default function LibraryPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent mb-8">
        Flashcard Library
      </h1>
      <FlashcardLibrary />
    </div>
  );
}
