// src/styles/HeaderStyles.ts
import { CSSProperties } from "react";

const styles: { [key: string]: CSSProperties } = {
  appHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    borderRadius: "0 0 1rem 1rem",
    // background: "rgb(var(--text-color))", // Adjust the background color as needed
    boxShadow: "0px 4px 4px 2px rgba(13, 13, 13, 0.29)",
    minWidth: "450px",
    position: "sticky",
    height: "80px",
  },
  wrapper: {
    maxWidth: "1250px",
    minWidth: "450px",

    margin: "20px",
  },
  logoContainer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-evenly",
    width: "20%",
  },
  appLogo: {
    height: "40px", // Adjust the logo size as needed
    marginRight: "12px",
  },
  appName: {
    fontSize: "24px",
    fontWeight: "bold",
    color: "#333333", // Adjust the text color as needed
  },
  gearIcon: {
    fontSize: "24px",
    color: "#333333", // Adjust the icon color as needed
    cursor: "pointer",
    // Note: Pseudo-classes like :hover can't be represented directly in inline styles.
    // You can handle hover effects using JavaScript events or a CSS-in-JS library that supports them.
  },
};

export default styles;
