import axios from "axios";
import { CreateCardInterface } from "./interfaces/api";

const BASE_URL = process.env.REACT_APP_BACKEND_URL;

// Ensure cookies are included in requests
axios.defaults.withCredentials = true;

export const createCard = (cardData: CreateCardInterface) => {
  console.log("Creating Card");

  return axios
    .post(`${BASE_URL}/create-card`, { data: cardData })
    .then((response) => {
      console.log("Successfully created card");
      return response.data;
    })
    .catch((error) => {
      console.error("Error creating card:", error);
      throw error;
    });
};
