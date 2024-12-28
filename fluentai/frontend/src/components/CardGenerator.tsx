import { useState, useEffect } from "react";
import AutoCompleteInput from "./ui/AutoCompleteInput";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import { createCard } from "../app/api/createCard";
import { AnkiService } from "@/services/anki/ankiService";
import { CreateCardInterface } from "../interfaces/CreateCardInterface";
import { getSupportedLanguages } from "@/app/api/languageService";
import SaveToAnki from "./SaveToAnki";
import { Card } from "@/interfaces/AnkiInterface";

interface CardGeneratorProps {
  onCardCreated: (card: Card) => void;
  onLoading: (loading: boolean) => void;
  onError: (error: string) => void;
  onWordChange: (word: string) => void;
}

const ankiService = new AnkiService();

export default function CardGenerator({
  onCardCreated,
  onLoading,
  onError,
  onWordChange,
}: CardGeneratorProps) {
  const [languages, setLanguages] = useState<{ [key: string]: string }>({});
  const [decks, setDecks] = useState<string[]>([]);
  const [input, setInput] = useState<CreateCardInterface>({
    language_code: "",
    word: "",
  });
  const [errors, setErrors] = useState({ language_code: "", word: "" });
  const [card, setCard] = useState<Card | null>(null);

  useEffect(() => {
    const fetchLanguagesAndDecks = async () => {
      try {
        onLoading(true);
        const [languageResponse, deckResponse] = await Promise.all([
          getSupportedLanguages(),
          ankiService.getAvailableDecks(),
        ]);
        setLanguages(languageResponse.languages);
        setDecks(deckResponse);
      } catch (error) {
        console.error("Error fetching data:", error);
        onError("Failed to load data.");
      } finally {
        onLoading(false);
      }
    };

    fetchLanguagesAndDecks();
  }, [onLoading, onError]);

  const validate = () => {
    const newErrors = {
      language_code: input.language_code ? "" : "Language is required.",
      word: input.word ? "" : "Word is required.",
    };
    setErrors(newErrors);
    return !newErrors.language_code && !newErrors.word;
  };

  const handleWordChange = (word: string) => {
    setInput((prev) => ({ ...prev, word }));
    onWordChange(word);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    onLoading(true);
    onError("");
    setCard(null);

    try {
      const response = await createCard(input);
      const newCard: Card = {
        img: response.imageUrl,
        word: input.word,
        keyPhrase: response.verbalCue,
        translation: response.translation,
        recording: response.ttsUrl,
        ipa: response.ipa,
      };
      setCard(newCard);
      onCardCreated(newCard);
    } catch (err: any) {
      onError(err.message || "An unexpected error occurred.");
    } finally {
      onLoading(false);
    }
  };

  return (
    <>
      <div className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:shadow-xl">
        <h2 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-teal-400 mb-6 text-center">
          Create Your Flashcard
        </h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <FormField
            label="Language"
            value={input.language_code}
            error={errors.language_code}
            required
          >
            <AutoCompleteInput
              suggestions={Object.keys(languages)}
              onSelect={(languageName) => {
                const languageCode =
                  languages[languageName as keyof typeof languages];
                setInput((prev) => ({
                  ...prev,
                  language_code: languageCode || "",
                }));
              }}
            />
          </FormField>
          <FormField
            label="Word"
            value={input.word}
            error={errors.word}
            required
            onChange={handleWordChange}
          />
          <Button
            text="Create Card"
            variant="primary"
            type="submit"
            className="w-full py-3 text-lg font-bold transform hover:scale-105 transition-transform duration-200 hover:shadow-lg"
          />
        </form>
      </div>
      {card && (
        <div className="bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:shadow-xl">
          <h3 className="text-xl font-bold mb-4">Save to Anki</h3>
          <SaveToAnki
            card={card}
            decks={decks}
            onError={onError}
            onLoading={onLoading}
          />
        </div>
      )}
    </>
  );
}
