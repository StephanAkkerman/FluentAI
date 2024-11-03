// src/pages/Header.tsx
import React from "react";
import styles from "../styles/HeaderStyles";

// Import the logo image
import logo from "../icons/logo.png";
import GearSvg from "../icons/svg/settings-gear";

type HeaderProps = {
  onGearClick: () => void;
};

const Header: React.FC<HeaderProps> = ({ onGearClick }) => {
  return (
    <header style={styles.appHeader}>
      <div className="wrapper" style={styles.wrapper}>
        <div className="logoContainer" style={styles.logoContainer}>
          <img src={logo} alt="FluentAI Logo" style={styles.appLogo} />
          <h1 style={styles.appName}>FluentAI</h1>
        </div>
        <div
          style={styles.gearIconWrapper}
          onClick={onGearClick}
          // Add event handlers for hover effect if needed
        >
          <GearSvg className="gear-icon" style={styles.gearIcon} />
        </div>
      </div>
    </header>
  );
};

export default Header;
