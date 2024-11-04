// src/pages/Header.tsx
import React, { useState } from "react";
import styles from "../styles/CardCreationStyles.module.css";

// components
import Button from "../components/button";
import AutocompleteInput from "../components/autocomplete";
import FormField from "../components/formfield";
import Card from "../components/card";

type CardCreationProps = {};

const CardCreation: React.FC<CardCreationProps> = () => {
  const handleSubmit = () => {};
  const handleLanguageSelect = () => {};
  const languages = ["Dutch", "English", "French", "Spanish"];
  const [inputWord, setinputWord] = useState("");
  return (
    <>
      <div className={styles.cardCreation}>
        <div className={styles.wrapper}>
          <div className={styles.formWrapper}>
            <h1 className={styles.title}>Card Creation</h1>
            <div className={styles.form}>
              <AutocompleteInput
                onSelect={handleLanguageSelect}
                suggestions={languages}
                title={"Select the language you want to learn!"}
              />
              <FormField
                value={inputWord}
                label={"Type word here"}
                id={"inputWord"}
                required={true}
                onChange={setinputWord}
                limit={50}
              />
            </div>
            <div className={styles.actions}>
              <Button
                text="Cancel"
                onClick={handleSubmit}
                style={{ cancel: true }}
              />
              <Button text="Submit" onClick={handleSubmit} />
            </div>
          </div>
          <Card />
        </div>
      </div>
    </>
  );
};

export default CardCreation;
