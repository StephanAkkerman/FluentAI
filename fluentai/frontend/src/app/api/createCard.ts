import axios from "axios";
import { CreateCardInterface, CreateCardResponse } from "../../interfaces/CreateCardInterface";

// API Endpoint
//const CARD_GEN_URL = process.env.REACT_APP_CARD_GEN_BACKEND_URL;
const CARD_GEN_URL = "http://localhost:8000";

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
      `${CARD_GEN_URL}/create_card/word_data`,
      cardData,
      { responseType: "json" }
    );

    const { IPA, recording } = wordData;

    // Step 2: Fetch image
    const { data: imgBlob } = await axios.get(
      `${CARD_GEN_URL}/create_card/img`,
      {
        params: {
          word: cardData.word,
          language_code: cardData.language_code,
        },
        responseType: "blob",
      }
    );

    // Step 3: Convert image blob to URL
    const imageUrl = URL.createObjectURL(imgBlob);

    const response: CreateCardResponse = { imageUrl, IPA, recording };

    console.log("Card created successfully:", response);
    return response;
  } catch (error: any) {
    console.error("Error creating card:", error.message || error);
    throw new Error(error.message || "Failed to create card.");
  }
};
