"use client";

import { useState } from "react";
import CardGenerator from "../components/CardGenerator";
import Flashcard from "../components/Flashcard";
import { Card } from "@/interfaces/CardInterfaces";
import StatusChecker from "@/components/StatusChecker";

export default function Home() {
  const [card, setCard] = useState<Card | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentWord, setCurrentWord] = useState("");

  const defaultCard: Card = {
    word: currentWord || "Your word",
    translation: "This is the English word.",
    verbalCue: "This is the key phrase",
    imageUrl: "https://placehold.co/400",
    audioUrl: "",
    ipa: "jʊər wɜrd",
    languageCode: "en"
  };

  return (
    <div className="flex flex-col gap-12">
      <StatusChecker />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
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
            card={card || defaultCard}
            isLoading={loading}
            showFront={!!card}
          />
        </div>
      </div>
    </div>
  );
}
