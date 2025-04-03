"use client"
import React, { useRef, useEffect, useState, ReactNode } from "react";
import Browser from "./ui/Browser";
import { motion, useScroll, useTransform, useInView } from "framer-motion";

const HowSection = () => {
    // Create a ref for the section
    const sectionRef = useRef(null);
    const [browserContent, setBrowserContent] = useState<ReactNode>(
        <>
            <div className="w-full flex items-center justify-center h-48">
                <h1 className="text-xl font-bold text-white">
                    Step 1: <br /> Create Your Card
                </h1>
            </div>
        </>
    );


    // Set up scroll progress tracking for this section only
    const { scrollYProgress } = useScroll({
        target: sectionRef
    });

    useEffect(() => {
        const unsubscribe = scrollYProgress.onChange(value => {
            if (value < 0.22) {
                setBrowserContent(
                    <div className="w-full flex items-center justify-center h-48">
                        <h1 className="text-xl font-bold text-white">
                            Step 1: <br /> Create Your Card
                        </h1>
                    </div>
                );
            } else if (value >= 0.22 && value < 0.5) {
                setBrowserContent(
                    <div className="w-full flex items-center justify-center h-48">
                        <h1 className="text-xl font-bold text-white">
                            Step 2: <br /> Customize Your Settings
                        </h1>
                    </div>
                );
            } else if (value >= 0.5 && value < 0.85) {
                setBrowserContent(
                    <div className="w-full flex items-center justify-center h-48">
                        <h1 className="text-xl font-bold text-white">
                            Step 3: <br /> Share With Your Team
                        </h1>
                    </div>
                );
            } else {
                setBrowserContent(
                    <div className="w-full flex items-center justify-center h-48">
                        <h1 className="text-xl font-bold text-white">
                            Step 4: <br /> Learn Via Flashcards
                        </h1>
                    </div>
                );
            }
        });
        return () => unsubscribe();
    }, [scrollYProgress]);




    // Create the transform animations based on scroll progress
    const transformAnimations = {
        browserTransform: useTransform(
            scrollYProgress,
            [0.22, 0.225, 0.5, 0.505, 0.85, 0.855],
            ["translate(10%, 0%)",
                "translate(45%, 0%)", "translate(45%, 0%)", "translate(10%, 0%)",
                "translate(10%, 0%)", "translate(30%, 50%)"]
        ),


        sectionOneTransform: useTransform(
            scrollYProgress,
            [0, 0.2],
            ["translate(60%, 0%)", "translate(60%, -600%)"]
        ),
        sectionOneOpacity: useTransform(
            scrollYProgress,
            [0, 0.03, 0.15, 0.2],
            ["0", "1", "1", "0"]
        ),
        sectionTwoTransform: useTransform(
            scrollYProgress,
            [0.2, 0.5],
            ["translate(0%, 0%)", "translate(0%, -775%)"]
        ),
        sectionTwoOpacity: useTransform(
            scrollYProgress,
            [0.23, 0.26, 0.5, 0.55],
            ["0", "1", "1", "0"]
        ),
        sectionThreeTransform: useTransform(
            scrollYProgress,
            [0.45, 0.75],
            ["translate(60%, 0%)", "translate(60%, -775%)"]
        ),
        sectionThreeOpacity: useTransform(
            scrollYProgress,
            [0.5, 0.51, 0.80, 0.85],
            ["0", "1", "1", "0"]
        )
    };


    return (
        <section
            ref={sectionRef}
            id="how-it-works"
            className="relative w-full h-[650vh]"
        >
            <div className="container mx-auto px-6 sticky top-[8rem]">
                <h2 className=" text-3xl font-bold text-center text-gray-800 mb-12">
                    How It Works
                </h2>

                {/* Only render the animated content when section is in view or has been in view */}
                <motion.div
                    style={{ transform: transformAnimations.browserTransform }}
                    className="transition-all duration-300"
                >

                    <Browser
                        urlText="https://FluentAI.com"
                        dark={true}
                        className="w-[50%] max-w-[500px]"
                    >
                        {browserContent}
                    </Browser>
                </motion.div>
                <motion.div
                    style={{ transform: transformAnimations.sectionOneTransform, opacity: transformAnimations.sectionOneOpacity }}
                    className="transition-all duration-300  ">
                    <div className="bg-gradient-to-r from-blue-500 to-teal-400 rounded-xl w-[40%] p-1">
                        <div className="bg-gray-50 rounded-xl p-2">
                            <h3 className="font-bold text-xl">Step 1</h3>
                            <p className="">First we will select a word we want to learn</p>
                        </div>
                    </div>

                </motion.div>
                <motion.div
                    style={{ transform: transformAnimations.sectionTwoTransform, opacity: transformAnimations.sectionTwoOpacity }}
                    className="transition-all duration-300 ">
                    <div className="bg-gradient-to-r from-blue-500 to-teal-400 rounded-xl w-[40%] p-1">
                        <div className="bg-gray-50 rounded-xl p-2">
                            <h3 className="font-bold text-xl">Step 2</h3>
                            <p className="">Create a card so we can start to learn</p>
                        </div>
                    </div>

                </motion.div>
                <motion.div
                    style={{ transform: transformAnimations.sectionThreeTransform, opacity: transformAnimations.sectionThreeOpacity }}
                    className="transition-all duration-300 ">
                    <div className="bg-gradient-to-r from-blue-500 to-teal-400 rounded-xl w-[40%] p-1">
                        <div className="bg-gray-50 rounded-xl p-2">
                            <h3 className="font-bold text-xl">Step 3</h3>
                            <p className="">Create a card so we can start to learn</p>
                        </div>
                    </div>

                </motion.div>
            </div>
        </section>
    );
};

export default HowSection;