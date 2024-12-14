"use client";

import { useState } from "react";
import CardGenerator from "../components/CardGenerator";
import Flashcard from "../components/Flashcard";

export default function Home() {
  const [card, setCard] = useState<{ img: string; word: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentWord, setCurrentWord] = useState("");

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start min-h-screen">
      <div>
        <CardGenerator
          onCardCreated={setCard}
          onLoading={setLoading}
          onError={setError}
          onWordChange={setCurrentWord}
        />
        {error && <p className="text-red-500 font-medium mt-4">{error}</p>}
      </div>
      <div className="flex items-center justify-center">
        <Flashcard
          word={currentWord}
          imageUrl={card?.img || ""}
          correctWord="This is the English word." // TODO: Get this from the API
          phrase="This is the key phrase" // TODO: Get this from the API
          isLoading={loading}
          showFront={!!card}
        />
      </div>
    </div>
  );
}
