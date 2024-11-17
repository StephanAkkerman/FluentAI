import axios from "axios";
import { CreateCardInterface } from "./interfaces/api";

const CARD_GEN_URL = process.env.REACT_APP_CARD_GEN_BACKEND_URL;

// Ensure cookies are included in requests
axios.defaults.withCredentials = true;

export const createCard = async (cardData: CreateCardInterface) => {
  console.log("Creating Card");

  try {
    const response = await axios.post(`${CARD_GEN_URL}/create_card`, cardData, {
      responseType: "json", // Expect a JSON response with metadata and image URL/stream
    });

    const { IPA, recording, image } = response.data;

    // Fetch the image as a stream using the image URL or directly from the image stream
    const imageResponse = await axios.get(image, { responseType: "blob" });

    // Convert the blob into a URL to display the image
    const imageUrl = URL.createObjectURL(imageResponse.data);

    console.log("Successfully created card");
    return {
      imageUrl,
      IPA,
      recording,
    };
  } catch (error) {
    console.error("Error creating card:", error);
    throw error;
  }
};
