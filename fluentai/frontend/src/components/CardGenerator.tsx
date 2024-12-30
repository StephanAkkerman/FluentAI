import { useState, useEffect } from "react";
import AutoCompleteInput from "./ui/AutoCompleteInput";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import { createCard } from "../app/api/createCard";
import { getSupportedLanguages } from "@/app/api/languageService";
import SaveToAnki from "./SaveToAnki";
import { Card, CreateCardRequest } from "@/interfaces/CardInterfaces";
import { ModelOptions } from "@/interfaces/ModelInterface";
import { ModelService } from "@/services/modelService";

interface CardGeneratorProps {
  onCardCreated: (card: Card) => void;
  onLoading: (loading: boolean) => void;
  onError: (error: string) => void;
  onWordChange: (word: string) => void;
}

const modelService = new ModelService();

export default function CardGenerator({
  onCardCreated,
  onLoading,
  onError,
  onWordChange,
}: CardGeneratorProps) {
  const [languages, setLanguages] = useState<{ [key: string]: string }>({});
  const [modelOptions, setModelOptions] = useState<ModelOptions>({
    imageModels: [],
    llmModels: []
  });
  const [input, setInput] = useState<CreateCardRequest>({
    languageCode: "",
    word: "",
    mnemonicKeyword: "",
    keySentence: "",
    imageModel: "",
    llmModel: ""
  });
  const [errors, setErrors] = useState({
    languageCode: "",
    word: ""
  });
  const [card, setCard] = useState<Card | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        onLoading(true);
        const [languageResponse, modelResponse] = await Promise.all([
          getSupportedLanguages(),
          modelService.getAvailableModels()
        ]);

        setLanguages(languageResponse.languages);
        setModelOptions(modelResponse);

        // Set default models if available
        if (modelResponse.imageModels.length > 0) {
          setInput(prev => ({ ...prev, imageModel: modelResponse.imageModels[0] }));
        }
        if (modelResponse.llmModels.length > 0) {
          setInput(prev => ({ ...prev, llmModel: modelResponse.llmModels[0] }));
        }
      } catch (error) {
        console.error("Error fetching data:", error);
        onError("Failed to load data.");
      } finally {
        onLoading(false);
      }
    };

    fetchData();
  }, [onLoading, onError]);

  const validate = () => {
    const newErrors = {
      languageCode: input.languageCode ? "" : "Language is required.",
      word: input.word ? "" : "Word is required.",
    };
    setErrors(newErrors);
    return !newErrors.languageCode && !newErrors.word;
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
      const newCard = await createCard(input);
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
            value={input.languageCode}
            error={errors.languageCode}
            required
          >
            <AutoCompleteInput
              suggestions={Object.keys(languages)}
              onSelect={(languageName) => {
                const languageCode = languages[languageName as keyof typeof languages];
                setInput((prev) => ({
                  ...prev,
                  languageCode: languageCode || "",
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

          <FormField
            label="Mnemonic Keyword (Optional)"
            value={input.mnemonicKeyword || ""}
            onChange={(mnemonicKeyword) => setInput(prev => ({ ...prev, mnemonicKeyword }))}
          />

          <FormField
            label="Key Sentence (Optional)"
            value={input.keySentence || ""}
            onChange={(keySentence) => setInput(prev => ({ ...prev, keySentence }))}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField
              label="Image Model"
              value={input.imageModel || ""}
            >
              <select
                className="w-full py-2 px-4 border rounded bg-white text-gray-800"
                value={input.imageModel}
                onChange={(e) => setInput(prev => ({ ...prev, imageModel: e.target.value }))}
              >
                {modelOptions.imageModels.map((model: string) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField
              label="LLM Model"
              value={input.llmModel || ""}
            >
              <select
                className="w-full py-2 px-4 border rounded bg-white text-gray-800"
                value={input.llmModel}
                onChange={(e) => setInput(prev => ({ ...prev, llmModel: e.target.value }))}
              >
                {modelOptions.llmModels.map((model: string) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </FormField>
          </div>

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
            onError={onError}
          />
        </div>
      )}
    </>
  );
}
