"use client";

import { useState } from "react";
import CardGenerator from "../components/CardGenerator";
import Flashcard from "../components/Flashcard";
import { Card } from "@/interfaces/AnkiInterface";

export default function Home() {
  const [card, setCard] = useState<Card | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentWord, setCurrentWord] = useState("");

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start min-h-screen">
      <div className="flex gap-12 flex-col">
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
          correctWord={card?.translation || "This is the English word."}
          phrase={card?.keyPhrase || "This is the key phrase"}
          isLoading={loading}
          showFront={!!card}
        />
      </div>
    </div>
  );
}
