import React, { useEffect, useState } from "react";

interface FlashcardProps {
  word: string;
  imageUrl: string;
  correctWord: string;
  phrase: string;
  isLoading: boolean;
  showFront?: boolean;
}

export default function Flashcard({
  word,
  imageUrl,
  correctWord,
  phrase,
  isLoading,
  showFront = false,
}: FlashcardProps) {
  const [flipped, setFlipped] = useState(!showFront);

  useEffect(() => {
    setFlipped(!showFront);
  }, [showFront]);

  useEffect(() => {
    if (isLoading) {
      setFlipped(false);
    } else if (imageUrl) {
      setFlipped(false);
    }
  }, [isLoading, imageUrl]);

  const toggleFlip = () => setFlipped(!flipped);

  return (
    <div
      className={`relative w-80 h-96 perspective cursor-pointer group`}
      onClick={toggleFlip}
    >
      <div
        className={`absolute inset-0 transform transition-transform duration-700 ease-in-out ${flipped ? "rotate-y-180" : ""}`}
        style={{ transformStyle: "preserve-3d" }}
      >
        {/* Front Side */}
        <div className="absolute inset-0 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 backface-hidden flex flex-col justify-center items-center p-6">
          {isLoading ? (
            <p className="text-blue-500 font-bold animate-pulse">Loading...</p>
          ) : (
            <>
              <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4 text-center">{correctWord}</h2>
              <div className="w-full h-64 overflow-hidden rounded-xl">
                <img
                  src={imageUrl || "https://placehold.co/400"}
                  alt={correctWord}
                  className="w-full h-full object-cover"
                />
              </div>
            </>
          )}
        </div>

        {/* Back Side */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-100 to-teal-100 dark:bg-gradient-to-br dark:from-blue-800 dark:to-teal-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 backface-hidden transform rotate-y-180 flex flex-col justify-center items-center p-6 text-center">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
            {word || "Your word"}
          </h2>
          <p className="text-lg italic text-gray-700 dark:text-gray-200">
            {phrase || "This is the key phrase"}
          </p>
        </div>
      </div>
    </div>
  );
}
