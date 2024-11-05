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
  const [isHovered, setIsHovered] = useState(false);
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
      if (isHovered) {
        // Stop animation and snap to nearest face
        api.stop();
        const currentRotation = rotationValueRef.current % 360;

        // Normalize rotation between -360 and 0
        const normalizedRotation =
          currentRotation < -360
            ? currentRotation + 360
            : currentRotation > 0
              ? currentRotation - 360
              : currentRotation;

        const targetRotation =
          normalizedRotation <= -90 && normalizedRotation >= -270
            ? -180
            : 0;

        api.start({ rotateY: targetRotation });
      } else {
        // Start infinite slow flipping
        api.start({
          loop: true,
          from: { rotateY: rotationValueRef.current },
          to: { rotateY: rotationValueRef.current - 360 },
          config: { duration: 1300 },
        });
      }
    } else {
      // Interactive mode: flip on click
      const targetRotation = flipped ? -180 : 0;
      api.start({ rotateY: targetRotation });
    }
  }, [showcase, isHovered, flipped, api]);

  const handleClick = () => {
    if (!showcase) {
      setFlipped((state) => !state);
    }
  };

  const handleMouseEnter = () => {
    if (showcase) {
      setIsHovered(true);
    }
  };

  const handleMouseLeave = () => {
    if (showcase) {
      setIsHovered(false);
    }
  };

  return (
    <div
      className={styles.container}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <a.div
        className={`${styles.c} ${styles.front}`}
        style={{
          opacity: rotation.rotateY.to((val) =>
            Math.max(0, Math.cos((val * Math.PI) / 180))
          ),
          transform: rotation.rotateY.to(
            (val) => `perspective(600px) rotateY(${val}deg)`
          ),
        }}
      >
        {front}
      </a.div>
      <a.div
        className={`${styles.c} ${styles.back}`}
        style={{
          opacity: rotation.rotateY.to((val) =>
            Math.max(0, Math.cos((val * Math.PI) / 180 + Math.PI))
          ),
          transform: rotation.rotateY.to(
            (val) => `perspective(600px) rotateY(${val + 180}deg)`
          ),
        }}
      >
        {back}
      </a.div>
    </div>
  );
};

export default Card;
