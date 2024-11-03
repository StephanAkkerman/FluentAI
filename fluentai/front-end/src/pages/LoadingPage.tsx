import React from "react";
import styles from "../styles/LoadingPageStyles";

// Import the logo image
import logo from "../icons/logo.png";

const LoadingPage: React.FC<{ isLoading: boolean }> = ({ isLoading }) => {
  const wrapperClass = isLoading
    ? "load-wrapper"
    : "load-wrapper load-wrapper-completed";

  return (
    <div className={wrapperClass} style={styles.loadWrapper}>
      <img src={logo} alt="FluentAI Logo" style={styles.appLogo} />
      <div className="loading-text" style={styles.loadingText}>
        Loading...
      </div>
    </div>
  );
};

export default LoadingPage;
