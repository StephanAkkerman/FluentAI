import { CSSProperties } from "react";

const styles: { [key: string]: CSSProperties } = {
  loadWrapper: {
    position: "fixed",
    zIndex: 1000,
    backgroundColor: "rgb(var(--second-color))",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    opacity: 1,
    transition: "opacity 0.5s ease-in-out, visibility 0.5s ease-in-out",
  },
  loadingText: {
    opacity: 0,
    background:
      "50% 100% / 50% 50% no-repeat radial-gradient(ellipse at bottom, rgb(var(--primary-color)), transparent, transparent)",
    WebkitBackgroundClip: "text",
    backgroundClip: "text",
    color: "transparent",
    fontWeight: 500,
    fontSize: "clamp(100px, 5vw, 125px)",
    fontFamily: "'Source Sans Pro', sans-serif",
    position: "relative",
    zIndex: 1,
  },
};

export default styles;
