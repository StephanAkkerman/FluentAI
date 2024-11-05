import { CSSProperties } from "react";

const styles: { [key: string]: CSSProperties } = {
  loadWrapper: {
    position: "fixed",
    zIndex: 1000,
    display: "flex",
    flexDirection: "column",
    backgroundColor: "rgb(var(--third-color))",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
    opacity: 1,
    transition: "opacity 0.5s ease-in-out, visibility 0.5s ease-in-out",
  },
  appLogo: {
    height: "250px",
    width: "250px",
    animation: "pulse 1.5s infinite",
  },
  loadingText: {
    marginTop: "10%",
    background:
      "50% 100% / 50% 50% no-repeat radial-gradient(ellipse at bottom, rgb(var(--primary-color)), transparent, transparent)",
    WebkitBackgroundClip: "text",
    backgroundClip: "text",
    color: "rgb(var(--text-color))",
    fontWeight: 500,
    fontSize: "clamp(16px, 5vw, 20px)",
    fontFamily: "'Source Sans Pro', sans-serif",
    position: "relative",
    zIndex: 1,
  },
};

// Define keyframes separately
const keyframes = `
  @keyframes pulse {
    0% {
      opacity: 1;
    }
    50% {
      opacity: 0.1;
    }
    100% {
      opacity: 1;
    }
  }
`;

// Inject keyframes into the document
const styleSheet = document.styleSheets[0];
styleSheet.insertRule(keyframes, styleSheet.cssRules.length);

export default styles;
