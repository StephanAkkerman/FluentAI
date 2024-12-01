// src/pages/Header.tsx
import React from "react";
import styles from "../styles/HeaderStyles.module.css";

// Import the logo image
import logo from "../icons/logo.png";
import GearSvg from "../icons/svg/settings-gear";

type HeaderProps = {
  onGearClick: () => void;
};

const Header: React.FC<HeaderProps> = ({ onGearClick }) => {
  return (
    <header className={styles.appHeader}>
      <div className={styles.wrapper}>
        <div className={styles.logoContainer}>
          <img src={logo} alt="FluentAI Logo" className={styles.appLogo} />
          <h1 className={styles.appName}>FluentAI</h1>
        </div>
        <div
          className={styles.gearIconWrapper}
          onClick={onGearClick}
        // Add event handlers for hover effect if needed
        >
          <GearSvg className={styles.gearIcon} />
        </div>
      </div>
    </header>
  );
};

export default Header;
