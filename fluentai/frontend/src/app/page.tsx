"use client";

import { useState } from "react";
import CardGenerator from "../components/CardGenerator";
import Flashcard from "../components/Flashcard";

export default function Home() {
  const [card, setCard] = useState<{ img: string; word: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  return (
    <div className="flex flex-row items-center justify-center h-screen gap-3">
      <div className="flex flex-col">
        <CardGenerator
          onCardCreated={setCard}
          onLoading={setLoading}
          onError={setError}
        />

        {loading && (
          <p className="text-blue-600 font-semibold mt-4">Creating your card...</p>
        )}

        {error && (
          <p className="text-red-500 font-semibold mt-4">{error}</p>
        )}
      </div>
      <div className="flex flex-col items-center justify-center h-screen">
        {card && (
          <div className="mt-6">
            <Flashcard
              word="This is the English word."
              imageUrl={card.img}
              correctWord={card.word}
              phrase="This is the key phrase"
            />
          </div>
        )}
      </div>
    </div>
  );
}
