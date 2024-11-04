import React, { useState } from "react";
import { useSpring, a } from "@react-spring/web";

import styles from "../styles/components/cardStyles.module.css";
import logo from "../icons/Logo (with stroke).png";

const Card = () => {
  const [flipped, set] = useState(false);
  const { transform, opacity } = useSpring({
    opacity: flipped ? 1 : 0,
    transform: `perspective(600px) rotateY(${flipped ? -180 : 0}deg)`,
    config: { mass: 5, tension: 500, friction: 80 },
  });
  return (
    <div className={styles.container} onClick={() => set((state) => !state)}>
      <a.div
        className={`${styles.c} ${styles.front}`}
        style={{ opacity: opacity.to((o) => 1 - o), transform }}
      >
        <h1>FluentAI</h1>
      </a.div>
      <a.div
        className={`${styles.c} ${styles.back}`}
        style={{
          opacity,
          transform,
          rotateY: "-180deg",
        }}
      >
        <img src={logo} alt="FluentAI Logo" />
      </a.div>
    </div>
  );
};

export default Card;
