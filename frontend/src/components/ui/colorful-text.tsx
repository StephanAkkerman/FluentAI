"use client";
import React from "react";
import { motion } from "framer-motion";

export function ColourfulText({ text }: { text: string }) {
    // Colors representing the blue-500 to teal-400 gradient
    const colors = [
        "rgb(59, 130, 246)",  // blue-500
        "rgb(56, 142, 239)",
        "rgb(54, 154, 232)",
        "rgb(52, 166, 225)",
        "rgb(50, 178, 218)",
        "rgb(48, 190, 211)",
        "rgb(46, 202, 204)",
        "rgb(45, 208, 197)",
        "rgb(45, 212, 191)",  // teal-400
        "rgb(45, 212, 191)",  // repeated to maintain 10 colors
    ];

    const [currentColors, setCurrentColors] = React.useState(colors);
    const [count, setCount] = React.useState(0);

    React.useEffect(() => {
        const interval = setInterval(() => {
            const shuffled = [...colors].sort(() => Math.random() - 0.5);
            setCurrentColors(shuffled);
            setCount((prev) => prev + 1);
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    return text.split("").map((char, index) => (
        <motion.span
            key={`${char}-${count}-${index}`}
            initial={{
                y: 0,
            }}
            animate={{
                color: currentColors[index % currentColors.length],
                y: [0, -3, 0],
                scale: [1, 1.01, 1],
                filter: ["blur(0px)", `blur(5px)`, "blur(0px)"],
                opacity: [1, 0.8, 1],
            }}
            transition={{
                duration: 0.5,
                delay: index * 0.05,
            }}
            className="inline-block whitespace-pre font-sans tracking-tight"
        >
            {char}
        </motion.span>
    ));
}