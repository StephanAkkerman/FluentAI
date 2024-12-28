import { useState, useEffect } from "react";
import AutoCompleteInput from "./ui/AutoCompleteInput";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import { createCard } from "../app/api/createCard";
import { AnkiService } from "@/services/anki/ankiService";
import { CreateCardInterface } from "../interfaces/CreateCardInterface";
import { getSupportedLanguages } from "@/app/api/languageService";

interface CardGeneratorProps {
  onCardCreated: (card: { img: string; word: string; keyPhrase: string; translation: string }) => void;
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
  const [selectedDeck, setSelectedDeck] = useState<string>(() => {
    // Load the default deck from localStorage (if available)
    return localStorage.getItem("selectedDeck") || "";
  });
  const [input, setInput] = useState<CreateCardInterface>({
    language_code: "",
    word: "",
  });
  const [errors, setErrors] = useState({ language_code: "", word: "" });
  const [card, setCard] = useState<{ img: string; word: string; keyPhrase: string; translation: string } | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

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

    try {
      const response = await createCard(input);
      const newCard = {
        img: response.imageUrl,
        word: input.word,
        keyPhrase: response.verbalCue,
        translation: response.translation,
      };
      setCard(newCard);
      onCardCreated(newCard);
    } catch (err: any) {
      onError(err.message || "An unexpected error occurred.");
    } finally {
      onLoading(false);
    }
  };

  const handleSaveToAnki = async () => {
    if (!card || !selectedDeck) {
      onError("Please select a deck.");
      return;
    }

    setSaveStatus('saving');
    onLoading(true);

    try {
      await ankiService.saveCard(card, selectedDeck);
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (error) {
      console.error("Error saving to Anki:", error);
      setSaveStatus('error');
      onError("Failed to save card to Anki.");
    } finally {
      onLoading(false);
    }
  };

  const handleDeckChange = (deck: string) => {
    setSelectedDeck(deck);
    // Save the selected deck to localStorage
    localStorage.setItem("selectedDeck", deck);
  };

  const getSaveButtonText = () => {
    switch (saveStatus) {
      case 'saving':
        return 'Saving...';
      case 'success':
        return 'Saved!';
      case 'error':
        return 'Failed to Save';
      default:
        return 'Save to Anki';
    }
  };

  const getSaveButtonVariant = () => {
    switch (saveStatus) {
      case 'success':
        return 'primary';
      case 'error':
        return 'danger';
      default:
        return 'secondary';
    }
  };

  return (
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
        {card && (
          <>
            <FormField label="Anki Deck" value={selectedDeck}>
              <select
                className="w-full py-2 px-4 border rounded"
                value={selectedDeck}
                onChange={(e) => handleDeckChange(e.target.value)}
              >
                <option value="" disabled>
                  Select a deck
                </option>
                {decks.map((deck) => (
                  <option key={deck} value={deck}>
                    {deck}
                  </option>
                ))}
              </select>
            </FormField>
            <Button
              text={getSaveButtonText()}
              variant={getSaveButtonVariant()}
              onClick={handleSaveToAnki}
              disabled={!selectedDeck || saveStatus === 'saving' || saveStatus === 'success'}
              className={`w-full py-3 text-lg font-bold ${saveStatus === 'saving' ? 'opacity-70 cursor-not-allowed' : ''
                }`}
            />
          </>
        )}
      </form>
    </div>
  );
}
