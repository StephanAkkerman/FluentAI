import { useState, useEffect } from "react";
import AutoCompleteInput from "./ui/AutoCompleteInput";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import { createCard } from "../app/api/createCard";
import { CreateCardInterface } from "../interfaces/CreateCardInterface";
import { getSupportedLanguages } from "@/app/api/languageService";

interface CardGeneratorProps {
  onCardCreated: (card: { img: string; word: string; keyPhrase: string; translation: string }) => void;
  onLoading: (loading: boolean) => void;
  onError: (error: string) => void;
  onWordChange: (word: string) => void;
}

export default function CardGenerator({
  onCardCreated,
  onLoading,
  onError,
  onWordChange,
}: CardGeneratorProps) {
  const [languages, setLanguages] = useState<{ [key: string]: string }>({});
  const [input, setInput] = useState<CreateCardInterface>({
    language_code: "",
    word: "",
  });
  const [errors, setErrors] = useState({ language_code: "", word: "" });

  useEffect(() => {
    const fetchLanguages = async () => {
      try {
        onLoading(true); // Indicate loading
        const response = await getSupportedLanguages();
        setLanguages(response.languages);
      } catch (error) {
        console.error("Error fetching languages:", error);
        onError("Failed to load supported languages.");
      } finally {
        onLoading(false); // Loading complete
      }
    };

    fetchLanguages();
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
      onCardCreated({ img: response.imageUrl, word: input.word, keyPhrase: response.verbalCue, translation: response.translation });
    } catch (err: any) {
      onError(err.message || "An unexpected error occurred.");
    } finally {
      onLoading(false);
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
      </form>
    </div>
  );
}

