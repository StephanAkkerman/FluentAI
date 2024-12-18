import { AnkiNote } from "../../interfaces/AnkiInterface";
import axios from "axios";

const ANKI_API_URL = "/FluentAI/api/anki";

export interface Card {
  img: string;
  word: string;
  keyPhrase: string;
  translation: string;
}

/**
 * Saves a generated card to Anki
 * @param card - The card data to be saved
 */
export const saveToAnki = async (card: Card): Promise<void> => {
  try {
    const note: AnkiNote = {
      deckName: "Model Deck",
      modelName: "Basic",
      fields: {
        Front: `${card.word}<br><img src="${card.img}" />`,
        Back: `${card.translation} - ${card.keyPhrase}`,
      },
      options: {
        allowDuplicate: false,
      },
      tags: ["FluentAI"],
    };

    await axios.post(ANKI_API_URL, {
      action: "addNote",
      version: 6,
      params: { note },
    });
  } catch (error) {
    console.error("Error saving to Anki:", error);
    throw new Error("Failed to save card to Anki.");
  }
};

/**
 * Fetches the list of available Anki decks
 */
export const getAvailableDecks = async (): Promise<string[]> => {
  try {
    const response = await axios.post(ANKI_API_URL, {
      action: "deckNames",
      version: 6,
    });
    return response.data.result;
  } catch (error) {
    console.error("Error fetching decks:", error);
    throw new Error("Failed to fetch decks.");
  }
};

/**
 * Loads cards from a specified Anki deck
 * @param deckName - The name of the deck to load cards from
 */
export const loadDeckCards = async (deckName: string): Promise<any[]> => {
  try {
    const response = await axios.post(ANKI_API_URL, {
      action: "findNotes",
      version: 6,
      params: {
        query: `deck:${deckName}`,
      },
    });

    const noteIds = response.data.result;

    const notesResponse = await axios.post(ANKI_API_URL, {
      action: "notesInfo",
      version: 6,
      params: {
        notes: noteIds,
      },
    });

    return notesResponse.data.result;
  } catch (error) {
    console.error("Error loading cards from deck:", error);
    throw new Error(`Failed to load cards from deck: ${deckName}`);
  }
};

