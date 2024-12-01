import axios from "axios";
import { CreateCardInterface } from "./interfaces/api";

const CARD_GEN_URL = process.env.REACT_APP_CARD_GEN_BACKEND_URL;

// Ensure cookies are included in requests
axios.defaults.withCredentials = true;

export const createCard = async (cardData: CreateCardInterface) => {
  console.log("Creating Card");

  try {
    // Step 1: Fetch word data (IPA and recording)
    const responseWordData = await axios.post(
      `${CARD_GEN_URL}/create_card/word_data`,
      cardData,
      {
        responseType: "json", // Expect a JSON response with metadata
      }
    );

    const { IPA, recording } = responseWordData.data;

    // Step 2: Fetch the image
    const responseImg = await axios.get(`${CARD_GEN_URL}/create_card/img`, {
      params: {
        word: cardData.word,
        language_code: cardData.language_code,
      },
      responseType: "blob", // Fetch image as a blob
    });

    // Step 3: Convert the blob into a URL to display the image
    const imageUrl = URL.createObjectURL(responseImg.data);

    const response = {
      imageUrl,
      IPA,
      recording,
    };

    console.log("Successfully created card", response);
    return response;
  } catch (error) {
    console.error("Error creating card:", error);
    throw error;
  }
};
