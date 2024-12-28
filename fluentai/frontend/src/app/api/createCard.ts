import axios from "axios";
import { Card, CreateCardRequest, createCardFromResponse } from '@/interfaces/CardInterfaces';

// API Endpoint
const CARD_GEN_URL = "http://localhost:8000/create_card";

// Ensure cookies are included
axios.defaults.withCredentials = true;

/**
 * Create a card by fetching word data and image
 */
export const createCard = async (request: CreateCardRequest): Promise<Card> => {
  try {
    console.log("Creating card...", request);

    // Step 1: Fetch all required data in a single API call
    const { data } = await axios.get(`${CARD_GEN_URL}/img`, {
      params: {
        word: request.word,
        language_code: request.languageCode,
      },
    });

    // Step 2: Convert base64 strings to Blob URLs
    const createBlobUrl = (base64Data: string, type: string): string => {
      const byteCharacters = atob(base64Data);
      const byteNumbers = Array.from(byteCharacters, char => char.charCodeAt(0));
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type });
      return URL.createObjectURL(blob);
    };

    const imageUrl = createBlobUrl(data.image, 'image/jpeg');
    const audioUrl = createBlobUrl(data.tts_file, 'audio/wav');

    // Step 3: Build response
    const responseWithUrls = { ...data, imageUrl, ttsUrl: audioUrl };

    // Step 4: Create and return the card
    return createCardFromResponse(request, responseWithUrls);
  } catch (error: any) {
    console.error("Error creating card:", error.message || error);
    throw new Error(error.message || "Failed to create card.");
  }
};
