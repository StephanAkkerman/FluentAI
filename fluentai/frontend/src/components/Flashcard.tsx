import React, { useEffect, useState, useRef } from "react";
import { Speaker } from "lucide-react";
import { Card } from "@/interfaces/AnkiInterface";

interface FlashcardProps {
  card: Card;
  isLoading: boolean;
  showFront?: boolean;
}

export default function Flashcard({
  card,
  isLoading,
  showFront = false,
}: FlashcardProps) {
  const [flipped, setFlipped] = useState(!showFront);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioError, setAudioError] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    setFlipped(!showFront);
  }, [showFront]);

  useEffect(() => {
    if (isLoading) {
      setFlipped(false);
    } else if (card.img) {
      setFlipped(false);
    }
  }, [isLoading, card.img]);

  // Set up audio event listeners
  useEffect(() => {
    const audioElement = audioRef.current;
    if (!audioElement) return;

    const handlePlay = () => setIsPlaying(true);
    const handleEnded = () => setIsPlaying(false);
    const handleError = () => {
      setAudioError(true);
      setIsPlaying(false);
      console.error("Audio playback error");
    };

    audioElement.addEventListener("play", handlePlay);
    audioElement.addEventListener("ended", handleEnded);
    audioElement.addEventListener("error", handleError);

    return () => {
      audioElement.removeEventListener("play", handlePlay);
      audioElement.removeEventListener("ended", handleEnded);
      audioElement.removeEventListener("error", handleError);
    };
  }, []);

  const toggleFlip = () => setFlipped(!flipped);

  const playAudio = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!audioRef.current || !card.recording) return;

    try {
      setAudioError(false);

      // Update audio source and play
      audioRef.current.src = card.recording;
      await audioRef.current.play();
    } catch (error) {
      console.error("Error playing audio:", error);
      setAudioError(true);
    }
  };

  return (
    <div
      className="relative w-80 h-96 perspective cursor-pointer group"
      onClick={toggleFlip}
    >
      <audio ref={audioRef} />
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
              <div className="w-full h-64 overflow-hidden rounded-xl">
                <img
                  src={card.img || "https://placehold.co/400"}
                  alt={card.word}
                  className="w-full h-full object-cover"
                />
              </div>
            </>
          )}
        </div>

        {/* Back Side */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-100 to-teal-100 dark:bg-gradient-to-br dark:from-blue-800 dark:to-teal-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 backface-hidden transform rotate-y-180 flex flex-col justify-center items-center p-6 text-center">
          <div className="w-full h-64 overflow-hidden rounded-xl mb-4">
            <img
              src={card.img || "https://placehold.co/400"}
              alt={card.word}
              className="w-full h-full object-cover"
            />
          </div>
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">
            {card.word || "Your word"}
          </h2>

          <div className="flex items-center gap-2 mb-2">
            <p className="font-mono text-lg text-gray-600 dark:text-gray-300">
              {card.ipa || "IPA pronunciation"}
            </p>
            {card.recording && (
              <button
                onClick={playAudio}
                className={`p-1 rounded-full transition-colors ${isPlaying
                  ? "bg-blue-100 dark:bg-blue-800"
                  : "hover:bg-gray-200 dark:hover:bg-gray-700"
                  } ${audioError
                    ? "text-red-500 hover:bg-red-100 dark:hover:bg-red-900"
                    : "text-gray-600 dark:text-gray-300"
                  }`}
                aria-label="Play pronunciation"
                disabled={isPlaying}
              >
                <Speaker className={`w-5 h-5 ${isPlaying ? "animate-pulse" : ""
                  }`} />
              </button>
            )}
          </div>

          <p className="text-lg italic text-gray-700 dark:text-gray-300">
            {card.keyPhrase || "This is the key phrase"}
          </p>
        </div>
      </div>
    </div>
  );
}
