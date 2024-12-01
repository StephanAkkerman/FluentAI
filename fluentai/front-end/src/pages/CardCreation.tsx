// src/pages/Header.tsx
import React, { useState } from "react";
import styles from "../styles/CardCreationStyles.module.css";

// components
import Button from "../components/button";
import AutocompleteInput from "../components/autocomplete";
import FormField from "../components/formfield";
import Card from "../components/card";

import logo from "../icons/Logo (with stroke).png";

import languages from "../config/languages.json";

import { createCard } from "../api";

type CardCreationProps = {};

const CardCreation: React.FC<CardCreationProps> = () => {
  const languagesMap: Record<string, string> = languages;
  const languagesArray = Object.keys(languagesMap); // User-friendly names
  const [input, setInput] = useState({
    language: "",
    word: "",
  });
  const [submit, setSubmitted] = useState(false);
  const [succes, setSucces] = useState({
    form: false,
    card: false,
  });
  const [card, setCard] = useState({
    img: "",
    word: "",
  });

  const FrontCardContent: React.FC = () => (
    <>
      {succes.card ? (
        <h1>{input.word}</h1>
      ) : succes.form ? (
        <h1>Loading</h1>
      ) : input.word ? (
        <h1>{input.word}</h1>
      ) : (
        <h1>FluentAI</h1>
      )}
    </>
  );

  const BackCardContent: React.FC = () => (
    <>
      {card.img ? (
        <img src={card.img} alt="Card Image" />
      ) : (
        <img src={logo} alt="FluentAI Logo" />
      )}
    </>
  );

  const handleSubmit = async () => {
    setSubmitted(true);
    if (input.language === "" || input.word === "") {
      return;
    }
    // TODO: handle supported language check
    setSucces((prevSucces) => ({
      ...prevSucces,
      form: true,
    }));

    try {
      // Call the API to create the card
      const response = await createCard({
        language_code: input.language,
        word: input.word,
      });

      console.log("Card created successfully:", response);

      // Update the card state with imageUrl and word
      setCard({
        img: response.imageUrl,
        word: input.word,
      });


      // Update the success state to show the card
      setSucces((prevSucces) => ({
        ...prevSucces,
        form: true,
        card: true,
      }));
    } catch (error) {
      console.error("Failed to create card:", error);

      // Optional: Display an error message to the user
      alert("Error creating card. Please try again.");
    }

  };

  const handleCancel = () => {
    setSubmitted(false);
    setSucces((prevSucces) => ({
      ...prevSucces,
      form: false,
      card: false,
    }));
  };

  const handleLanguage = (selectedLanguage: string) => {
    const languageCode = languagesMap[selectedLanguage]; // Get the language code
    setInput((prevInput) => ({
      ...prevInput,
      language: languageCode || "", // Set the code or an empty string if not found
    }));
    setSubmitted(false);
  };

  const handleWord = (value: string) => {
    setInput((prevInput) => ({
      ...prevInput,
      word: value,
    }));
    setSubmitted(false);
  };
  return (

    <div className={styles.cardCreation}>
      <div className={styles.wrapper}>



        <>
          <div className={styles.formWrapper}>
            <h1 className={styles.title}>Card Creation</h1>

            <div className={succes.form ? styles.formHidden : styles.form} onFocus={() => setSubmitted(false)}>
              <AutocompleteInput
                onSelect={handleLanguage}
                suggestions={languagesArray}
                title={"Select the language you want to learn!"}
                formSubmitted={submit}
              />
              <FormField
                value={input.word}
                label={"Type word here"}
                id={"inputWord"}
                required={true}
                onChange={handleWord}
                limit={50}
                formSubmitted={submit}
              />

            </div>
            <div
              className={succes.form && !succes.card ? styles.loading : styles.loadingHidden}
            >
              <h2>Creating your card in a Flash...</h2>
            </div>

            <div
              className={succes.card ? styles.loading : styles.loadingHidden}
            >
              <h2>Card Created Succesfully!</h2>
            </div>

            <div className={styles.actions}>
              <Button
                text={succes.card ? "View" : "Cancel"}
                onClick={handleCancel}
                style={{ cancel: true }}
              />
              {succes.form && !succes.card ? null : <Button text={succes.card ? "Create Another" : "Submit"} onClick={handleSubmit} />}

            </div>

          </div>
          <div className={styles.card}>
            <Card
              front={<FrontCardContent />}
              back={<BackCardContent />}
              showcase={succes.form && !succes.card ? true : false}
            />

          </div>
        </>

      </div>
    </div>
  )
};


export default CardCreation;
