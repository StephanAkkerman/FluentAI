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

type CardCreationProps = {};

const CardCreation: React.FC<CardCreationProps> = () => {
  const languagesArray = Object.keys(languages);
  const [input, setInput] = useState({
    language: "",
    word: "",
  });
  const [submit, setSubmitted] = useState(false);
  const [succes, setSucces] = useState({
    form: false,
    card: false,
  });

  const FrontCardContent: React.FC = () => (
    <>{input.word ? <h1>{input.word}</h1> : <h1>FluentAI</h1>}</>
  );

  const BackCardContent: React.FC = () => (
    <>
      <img src={logo} alt="FluentAI Logo" />
    </>
  );

  const handleSubmit = () => {
    setSubmitted(true);
    if (input.language === "" || input.word === "") {
      return;
    }
    // TODO: handle supported language check
    setSucces((prevSucces) => ({
      ...prevSucces,
      form: true,
    }));
  };

  const handleCancel = () => {
    setSubmitted(false);
    setSucces((prevSucces) => ({
      ...prevSucces,
      form: false,
      card: false,
    }));
  };

  const handleLanguage = (value: string) => {
    setInput((prevInput) => ({
      ...prevInput,
      language: value,
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
    <>
      <div className={styles.cardCreation}>
        <div className={styles.wrapper}>
          <div className={styles.formWrapper}>
            <h1 className={styles.title}>Card Creation</h1>
            <div className={styles.form} onFocus={() => setSubmitted(false)}>
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
            <div className={styles.actions}>
              <Button
                text="Cancel"
                onClick={handleCancel}
                style={{ cancel: true }}
              />
              <Button text="Submit" onClick={handleSubmit} />
            </div>
          </div>
          <div className={styles.card}>
            <Card
              front={<FrontCardContent />}
              back={<BackCardContent />}
              showcase={succes.form ? true : false}
            />
            <div
              className={succes.form ? styles.loadingHidden : styles.loading}
            >
              <h2>Creating your card in a Flash...</h2>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default CardCreation;
