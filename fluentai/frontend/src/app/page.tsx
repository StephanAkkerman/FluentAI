"use client";

import { useState } from "react";
import CardGenerator from "../components/CardGenerator";
import Flashcard from "../components/Flashcard";

export default function Home() {
  const [card, setCard] = useState<{ img: string; word: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start min-h-screen">
      <div>
        <CardGenerator
          onCardCreated={setCard}
          onLoading={setLoading}
          onError={setError}
        />
        {loading && (
          <p className="text-blue-600 font-medium mt-4">Creating your card...</p>
        )}
        {error && <p className="text-red-500 font-medium mt-4">{error}</p>}
      </div>
      <div className="flex items-center justify-center">
        {card && (
          <div>
            <Flashcard
              word="This is the English word."
              imageUrl={card.img}
              correctWord={card.word}
              phrase="This is the key phrase"
              showFront={false}
            />
          </div>
        )}
      </div>
    </div>
  );
}

