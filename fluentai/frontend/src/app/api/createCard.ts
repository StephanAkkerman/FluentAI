import axios from "axios";
import { CreateCardInterface, CreateCardResponse } from "../../interfaces/CreateCardInterface";

// API Endpoint
const CARD_GEN_URL = "http://localhost:8000/create_card";

// Ensure cookies are included
axios.defaults.withCredentials = true;

/**
 * Create a card by fetching word data and image
 */
export const createCard = async (
  cardData: CreateCardInterface
): Promise<CreateCardResponse> => {
  try {
    console.log("Creating card...", cardData);

    // Step 1: Fetch word data (IPA and recording)
    const { data: wordData } = await axios.post(
      `${CARD_GEN_URL}/word_data`,
      cardData,
      { responseType: "json" }
    );
    const { IPA, recording } = wordData;

    // Step 2: Fetch image with verbal cue and translation
    const { data } = await axios.get(
      `${CARD_GEN_URL}/img`,
      {
        params: {
          word: cardData.word,
          language_code: cardData.language_code,
        },
      }
    );

    // Convert base64 image to blob
    const imageBlob = await (await fetch(`data:image/jpeg;base64,${data.image}`)).blob();
    const imageUrl = URL.createObjectURL(imageBlob);

    const response: CreateCardResponse = {
      imageUrl,
      IPA,
      recording,
      verbalCue: data.verbal_cue,
      translation: data.translation
    };

    console.log("Card created successfully:", response);
    return response;
  } catch (error: any) {
    console.error("Error creating card:", error.message || error);
    throw new Error(error.message || "Failed to create card.");
  }
};
