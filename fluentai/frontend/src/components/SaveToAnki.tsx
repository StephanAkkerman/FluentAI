import React, { useState } from "react";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import { AnkiService } from "@/services/anki/ankiService";

interface SaveToAnkiProps {
  card: { img: string; word: string; keyPhrase: string; translation: string };
  decks: string[];
  onError: (error: string) => void;
  onLoading: (loading: boolean) => void;
}

const ankiService = new AnkiService();

export default function SaveToAnki({ card, decks, onError, onLoading }: SaveToAnkiProps) {
  const [selectedDeck, setSelectedDeck] = useState<string>("");
  const [testSpelling, setTestSpelling] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  const handleSaveToAnki = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedDeck) {
      onError("Please select a deck.");
      return;
    }

    setSaveStatus('saving');
    onLoading(true);

    try {
      await ankiService.saveCard(card, selectedDeck, testSpelling);
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
    if (typeof window !== "undefined") {
      localStorage.setItem("selectedDeck", deck);
    }
  };

  const handleSpellingPreferenceChange = (checked: boolean) => {
    setTestSpelling(checked);
    if (typeof window !== "undefined") {
      localStorage.setItem("testSpelling", checked.toString());
    }
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
    <form onSubmit={handleSaveToAnki} className="space-y-6">
      <FormField label="Anki Deck" value={selectedDeck} required>
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
      <FormField value="true">
        <div className="flex items-center">
          <input
            type="checkbox"
            id="testSpellingToggle"
            className="mr-2"
            checked={testSpelling}
            onChange={(e) => handleSpellingPreferenceChange(e.target.checked)}
          />
          <label htmlFor="testSpellingToggle">Enable Test Spelling</label>
        </div>
      </FormField>
      <Button
        text={getSaveButtonText()}
        variant={getSaveButtonVariant()}
        type="submit"
        disabled={!selectedDeck || saveStatus === 'saving' || saveStatus === 'success'}
        className={`w-full py-3 text-lg font-bold ${saveStatus === 'saving' ? 'opacity-70 cursor-not-allowed' : ''}`}
      />
    </form>
  );
}

