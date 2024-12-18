import axios from "axios";
import { SupportedLanguagesResponse } from "@/interfaces/LanguageInterface";

// API Endpoint
const CARD_GEN_URL = "http://localhost:8000/create_card";

// Ensure cookies are included
axios.defaults.withCredentials = true;

export const getSupportedLanguages = async (): Promise<SupportedLanguagesResponse> => {
  try {
    const result = await axios.get<SupportedLanguagesResponse>(
      `${CARD_GEN_URL}/supported_languages`
    );
    return result.data;
  } catch (error) {
    console.error("Error fetching supported languages:", error);
    throw new Error("Failed to fetch supported languages.");
  }
};
