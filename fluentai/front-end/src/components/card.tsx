import React, { useState, useEffect, useRef } from "react";
import { useSpring, a } from "@react-spring/web";
import styles from "../styles/components/cardStyles.module.css";

interface CardProps {
  front: React.ReactNode;
  back: React.ReactNode;
  showcase?: boolean;
}

const Card: React.FC<CardProps> = ({ front, back, showcase = false }) => {
  const [flipped, setFlipped] = useState(false);

  const rotationValueRef = useRef(0);

  const [rotation, api] = useSpring(() => ({
    rotateY: 0,
    config: { mass: 5, tension: 500, friction: 80 },
    onChange: ({ value }) => {
      rotationValueRef.current = value.rotateY;
    },
  }));

  // Handle flipping animation based on props
  useEffect(() => {
    if (showcase) {
      // Start infinite slow flipping
      api.start({
        loop: true,
        from: { rotateY: rotationValueRef.current },
        to: { rotateY: rotationValueRef.current - 360 },
        config: { duration: 1300 },
      });
    } else {
      // Interactive mode: flip on click
      const targetRotation = flipped ? -180 : 0;
      api.start({
        rotateY: targetRotation,
        config: { mass: 5, tension: 500, friction: 80 },
      });
    }
  }, [showcase, flipped, api]);

  const handleClick = () => {
    if (!showcase) {
      setFlipped((state) => !state);
    }
  };

  return (
    <div className={styles.container} onClick={handleClick}>
      <a.div
        className={`${showcase ? styles.cShowcase : styles.c} ${styles.front}`}
        style={{
          opacity: rotation.rotateY.to((val) =>
            Math.max(0, Math.cos((val * Math.PI) / 180))
          ),
          transform: rotation.rotateY.to(
            (val) => `perspective(600px) rotateY(${val}deg)`
          ),
        }}
      >
        <div className={styles.frontContent}>{front}</div>
      </a.div>
      <a.div
        className={`${showcase ? styles.cShowcase : styles.c} ${styles.back}`}
        style={{
          opacity: rotation.rotateY.to((val) =>
            Math.max(0, Math.cos((val * Math.PI) / 180 + Math.PI))
          ),
          transform: rotation.rotateY.to(
            (val) => `perspective(600px) rotateY(${val + 180}deg)`
          ),
        }}
      >
        <div className={styles.backContent}>{back}</div>
      </a.div>
    </div>
  );
};

export default Card;
