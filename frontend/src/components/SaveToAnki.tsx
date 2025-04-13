import React, { useState, useEffect, useCallback } from "react";
import FormField from "./ui/FormField";
import Button from "./ui/Button";
import { AnkiService } from "@/services/anki/ankiService";
import { Card } from "@/interfaces/CardInterfaces";
import { RefreshCw, Plus } from "lucide-react";

interface SaveToAnkiProps {
  card: Card;
  onError: (error: string) => void;
}

const ankiService = new AnkiService();

export default function SaveToAnki({ card, onError }: SaveToAnkiProps) {
  const [selectedDeck, setSelectedDeck] = useState<string>("");
  const [testSpelling, setTestSpelling] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [localLoading, setLocalLoading] = useState(false);
  const [decks, setDecks] = useState<string[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchDecks = useCallback(async () => {
    try {
      setIsRefreshing(true);
      const deckResponse = await ankiService.getAvailableDecks();
      setDecks(deckResponse);
    } catch (error) {
      console.error("Error fetching data:", error);
      onError("Failed to load data.");
    } finally {
      setIsRefreshing(false);
    }
  }, [onError]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setSelectedDeck(localStorage.getItem("selectedDeck") || "");
      setTestSpelling(localStorage.getItem("testSpelling") === "true");
    }

    fetchDecks();
  }, [fetchDecks]);

  const handleSaveToAnki = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedDeck) {
      onError("Please select a deck.");
      return;
    }

    setSaveStatus('saving');
    setLocalLoading(true);

    try {
      await ankiService.saveCard(card, selectedDeck, testSpelling);
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (error) {
      console.error("Error saving to Anki:", error);
      setSaveStatus('error');
      onError("Failed to save card to Anki.");
    } finally {
      setLocalLoading(false);
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

  const handleCreateDeck = async () => {
    const newDeckName = prompt("Enter the name of the new deck:");
    if (!newDeckName) return;

    try {
      await ankiService.createDeck(newDeckName);
      setDecks((prevDecks) => [...prevDecks, newDeckName]);
      setSelectedDeck(newDeckName);
      if (typeof window !== "undefined") {
        localStorage.setItem("selectedDeck", newDeckName);
      }
    } catch (error) {
      console.error("Error creating new deck:", error);
      onError("Failed to create new deck.");
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
        <div className="flex gap-2">
          <select
            className="w-full p-3 border rounded-lg bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 
                       focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
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
          <button
            type="button"
            onClick={handleCreateDeck}
            className="p-2 border rounded hover:bg-gray-100 dark:hover:bg-gray-900"
          >
            <Plus className="h-5 w-5" />
          </button>
          <button
            type="button"
            onClick={fetchDecks}
            disabled={isRefreshing}
            className="p-2 border rounded hover:bg-gray-100 dark:hover:bg-gray-900 disabled:opacity-50"
          >
            <RefreshCw
              className={`h-5 w-5 ${isRefreshing ? 'animate-spin' : ''}`}
            />
          </button>
        </div>
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
        text={localLoading ? "Saving..." : getSaveButtonText()}
        variant={getSaveButtonVariant()}
        type="submit"
        disabled={localLoading || !selectedDeck || saveStatus === "success"}
        className={`w-full py-3 text-lg font-bold ${saveStatus === 'saving' ? 'opacity-70 cursor-not-allowed' : ''}`}
      />
    </form>
  );
}
