import React, { useState } from "react";

interface FlashcardProps {
  word: string;
  imageUrl: string;
  correctWord: string;
  phrase: string;
}

export default function Flashcard({
  word,
  imageUrl,
  correctWord,
  phrase,
}: FlashcardProps) {
  const [flipped, setFlipped] = useState(false);

  const toggleFlip = () => setFlipped(!flipped);

  return (
    <div
      className="relative w-80 h-96 perspective cursor-pointer"
      onClick={toggleFlip}
    >
      <div
        className={`absolute inset-0 transform transition-transform duration-500 ${flipped ? "rotate-y-180" : ""
          }`}
        style={{ transformStyle: "preserve-3d" }}
      >
        {/* Front Side */}
        <div className="absolute inset-0 bg-white dark:bg-gray-800 rounded-2xl shadow-lg backface-hidden flex flex-col justify-center items-center p-4">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">
            {word}
          </h2>
          <img
            src={imageUrl}
            alt={word}
            className="w-full h-full object-cover rounded-md"
          />
        </div>

        {/* Back Side */}
        <div className="absolute inset-0 bg-white dark:bg-gray-800 rounded-2xl shadow-lg backface-hidden transform rotate-y-180 flex flex-col justify-center items-center p-4">
          <h2 className="text-2xl font-bold mb-2">{correctWord}</h2>
          <p className="text-lg italic text-gray-800 dark:text-gray-200 text-center">{phrase}</p>
        </div>
      </div>
    </div>
  );
}

