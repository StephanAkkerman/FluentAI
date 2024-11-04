import { CSSProperties } from "react";

const styles: { [key: string]: CSSProperties } = {
  cardCreation: {
    display: "flex",
    flexDirection: "column",
    top: 0,
    left: 0,
    width: "100%",
    height: "90vh",
    alignItems: "center",
    justifyContent: "center",
    opacity: 1,
    transition: "opacity 0.5s ease-in-out, visibility 0.5s ease-in-out",
  },
  wrapper: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    top: 0,
    left: 0,
    width: "90%",
    height: "50%",
    minHeight: "450px",
    maxWidth: "900px",
    maxHeight: "800px",
    background: "rgb(var(--second-color))",
    borderRadius: "5rem",
    boxShadow: "0px 4px 4px 2px rgba(13, 13, 13, 0.29)",
  },
  form: {
    height: "85%",
  },
  actions: {
    width: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-evenly",
  },
};

export default styles;
