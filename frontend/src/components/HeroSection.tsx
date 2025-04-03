"use client"
import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { motion, useScroll, useTransform } from "framer-motion";
import duck from "../../public/duck.jpg";

interface FlashcardProps {
    isFlipped: boolean;
    handleFlip: () => void;
}

const HeroSection = () => {
    // Create a ref for the section
    const sectionRef = useRef(null);
    const { scrollYProgress } = useScroll({
        target: sectionRef
    });
    const [isFlipped, setIsFlipped] = useState(false);
    const transformAnimations = {
        h1Transform: useTransform(
            scrollYProgress,
            [0, 0.3, 0.8, 1],
            ["translate(8%, -30%)", "translate(-20%, 0%)", "translate(-20%, 0%)", "translate(-20%, -10%)"]
        ),
        h2Transform: useTransform(
            scrollYProgress,
            [0, 0.3, 0.8, 1],
            ["translate(15%, -30%)", "translate(-20%, 0%)", "translate(-20%, 0%)", "translate(-20%, -10%)"]
        ),
        cardTransform: useTransform(
            scrollYProgress,
            [0, 0.3, 0.8, 1],
            ["translate(-50%, 60%)", "translate(20%, 20%)", "translate(20%, 20%)", "translate(20%, 10%)"]
        ),

        titleFontSize: useTransform(
            scrollYProgress,
            [0, 0.3],
            ["2.5rem", "2rem"]
        ),
        subtitleFontSize: useTransform(
            scrollYProgress,
            [0, 0.3],
            ["2.25rem", "1.5rem"]
        ),
        contentOpacity: useTransform(scrollYProgress, [0, 0.3], [0, 1]),
        bgOpacity: useTransform(scrollYProgress, [0, 0.3], [0.3, 0.7]),
        bgWidth: useTransform(scrollYProgress, [0, 0.3, 0.8, 0.9], ["50%", "100%", "100%", "95%"]),
        bgHeight: useTransform(scrollYProgress, [0, 0.3, 0.8, 0.9], ["100%", "100%", "100%", "80%"])
    }


    useEffect(() => {
        const unsubscribe = scrollYProgress.on("change", (latest) => {
            setIsFlipped(latest > 0.1);
        });

        return () => unsubscribe();
    }, [scrollYProgress]);

    // For handling manual flip when not scrolling
    const handleFlip = () => {
        setIsFlipped(!isFlipped);
    };


    return (
        <div ref={sectionRef} className="relative w-full h-[350vh] ">
            {/* This is the container that stays fixed in the viewport */}
            <div className="sticky top-0 h-screen w-full overflow-hidden flex justify-center">
                {/* Background */}
                <motion.div
                    style={{ opacity: transformAnimations.bgOpacity, width: transformAnimations.bgWidth, height: transformAnimations.bgHeight }}
                    className=" inset-0 h-[70%] bg-[#97E2F9] rounded-xl transition-all duration-300"
                />

                {/* Content Container */}
                <div className="absolute inset-0 w-full h-full flex items-center justify-center">
                    {/* Text Content - positioned absolute */}
                    <motion.div
                        className="absolute top-[20%] w-full max-w-lg transition-all duration-300"

                    >
                        {/* The title starts big and shrinks */}
                        <motion.h1
                            className="relative font-bold text-gray-800 mb-4 "
                            style={{ fontSize: transformAnimations.titleFontSize, transform: transformAnimations.h1Transform }}
                        >
                            Learn Languages Faster
                        </motion.h1>

                        <motion.h2
                            className="font-bold text-[#FF8A5B] mb-6 "
                            style={{ fontSize: transformAnimations.subtitleFontSize, transform: transformAnimations.h2Transform }}
                        >
                            Using AI Mnemonics
                        </motion.h2>

                        <motion.div style={{ opacity: transformAnimations.contentOpacity, transform: transformAnimations.h2Transform }}>
                            <p className="text-gray-600 mb-8 text-lg w-[70%]">
                                Create personalized flashcards with powerful memory techniques that make
                                learning engaging and effective.
                            </p>
                            <div className="flex flex-wrap gap-4">
                                <button className="px-8 py-3 rounded-full bg-[#FF8A5B] text-white font-medium hover:bg-[#ff7a43] transition-colors">
                                    Get Started
                                </button>
                                <button className="px-8 py-3 rounded-full bg-white border-2 border-[#FF8A5B] text-[#FF8A5B] font-medium hover:bg-[#fff8f6] transition-colors">
                                    See Demo
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* Flashcard container - positioned absolute */}
                    <motion.div
                        className="absolute left-1/2 top-[10%] transition-all duration-300"
                        style={{ transform: transformAnimations.cardTransform }}
                    >
                        {/* Flashcard that flips during scroll */}
                        <FlashCard isFlipped={isFlipped} handleFlip={handleFlip} />
                    </motion.div>
                </div>
            </div>
        </div>
    );
};

const FlashCard = ({ isFlipped, handleFlip }: FlashcardProps) => {
    return (
        <div className="relative w-80 h-96" style={{ perspective: "1000px" }}>
            <motion.div
                animate={{ rotateY: isFlipped ? 180 : 0 }}
                transition={{ duration: 0.7 }}
                className="absolute w-full h-full bg-gradient-to-r from-blue-500 to-teal-400 rounded-xl shadow-lg border-2 border-[#97E2F9] cursor-pointer"
                onClick={handleFlip}
                style={{ transformStyle: "preserve-3d" }}
            >
                {/* Front of card */}
                <div
                    className="absolute w-full h-full p-6 flex flex-col items-center justify-center"
                    style={{ backfaceVisibility: "hidden" }}
                >
                    <h3 className="text-2xl font-bold text-gray-800 mb-2">Pato</h3>
                </div>
                {/* Back of card */}
                <div
                    className="absolute w-full h-full p-6 flex flex-col items-center justify-center"
                    style={{ backfaceVisibility: "hidden", transform: "rotateY(180deg)" }}
                >
                    <Image
                        src={duck}
                        alt="duck"
                        width={250}
                        height={250}
                        className="rounded-xl"
                    />
                </div>
            </motion.div>
        </div>
    );
};

export default HeroSection;