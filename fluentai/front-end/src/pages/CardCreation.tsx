// src/pages/Header.tsx
import React from "react";
import styles from "../styles/CardCreationStyles";

// components
import Button from "../components/button";


type CardCreationProps = {

};

const CardCreation: React.FC<CardCreationProps> = () => {

    const handleSubmit = () => {

    }
    return (
        <div className="card-creation" style={styles.cardCreation}>
            <h1>Card Creation</h1>
            <div className="wrapper" style={styles.wrapper}>
                <div className="form" style={styles.form}>

                </div>
                <div className="actions" style={styles.actions}>
                    <Button text="Cancel" onClick={handleSubmit} style={{ cancel: true }} />
                    <Button text="Submit" onClick={handleSubmit} />

                </div>

            </div>

        </div>

    );
};

export default CardCreation;
