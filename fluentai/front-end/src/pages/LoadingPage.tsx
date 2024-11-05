import React from "react";
import styles from "../styles/LoadingPageStyles";

// Import the logo image
import logo from "../icons/Logo (with stroke).png";

import Card from "../components/card";

const LoadingPage: React.FC<{ isLoading: boolean }> = ({ isLoading }) => {
  const wrapperClass = isLoading
    ? "load-wrapper"
    : "load-wrapper load-wrapper-completed";

  const FrontCardContent: React.FC = () => (
    <div>
      <h1>FluentAI</h1>
    </div>
  );

  const BackCardContent: React.FC = () => (
    <div>
      <img src={logo} alt="FluentAI Logo" />
    </div>
  )


  return (
    <div className={wrapperClass} style={styles.loadWrapper}>

      <Card front={<FrontCardContent />} back={<BackCardContent />} showcase={true} />
      <div className="loading-text" style={styles.loadingText}>
        <h1>Loading...</h1>
      </div>
    </div>
  );
};

export default LoadingPage;
