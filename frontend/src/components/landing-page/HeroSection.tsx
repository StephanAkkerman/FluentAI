"use client"
import React, { useState, useEffect, useRef } from "react";
import Button from "../ui/Button";
import { ContainerTextFlip } from "../ui/container-text-flip";
import { motion, useInView, useScroll, useTransform } from "framer-motion";
import { SparklesCore } from "../ui/sparkles";
import Flashcard from "../Flashcard";
import duck from "../../../public/duck.jpg";

const HeroSection = () => {
  // Create a ref for the section
  const sectionRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef
  });
  const isInView = useInView(sectionRef, { once: true });
  const [isFlipped, setIsFlipped] = useState(false);
  // Add state to track if we're past the transition threshold
  const [isPastThreshold, setIsPastThreshold] = useState(false);

  const cardData = {
    word: "Pato",
    imageUrl: duck.src,
    audioUrl: "",
    ipa: "/ˈpɑtoʊ/",
    verbalCue: "Spanish word for 'duck'",
    translation: "Duck",
    languageCode: "es"
  };

  const transformAnimations = {
    h1Transform: useTransform(
      scrollYProgress,
      [0, 0.3, 0.8, 0.9],
      ["translate(4%, -30%)", "translate(-25%, 40%)", "translate(-25%, 40%)", "translate(-25%, 0%)"]
    ),
    h2Transform: useTransform(
      scrollYProgress,
      [0, 0.3, 0.8, 0.9],
      ["translate(15%, -30%)", "translate(-25%, 220%)", "translate(-25%, 220%)", "translate(-25%, 0%)"]
    ),
    pTransform: useTransform(
      scrollYProgress,
      [0, 0.3, 0.8, 0.9],
      ["translate(15%, -30%)", "translate(-25%, 60%)", "translate(-25%, 60%)", "translate(-25%, 20%)"]
    ),
    cardTransform: useTransform(
      scrollYProgress,
      [0, 0.3, 0.8, 0.9],
      ["translate(-50%, 60%)", "translate(30%, 50%)", "translate(30%, 50%)", "translate(30%, 15%)"]
    ),

    titleFontSize: useTransform(
      scrollYProgress,
      [0, 0.3],
      ["2.5rem", "3rem"]
    ),
    subtitleFontSize: useTransform(
      scrollYProgress,
      [0, 0.3],
      ["2.25rem", "1.5rem"]
    ),
    contentOpacity: useTransform(scrollYProgress, [0, 0.3], [0, 1]),
    bgOpacity: useTransform(scrollYProgress, [0, 0.3], [0.3, 0.7]),
    bgWidth: useTransform(scrollYProgress, [0.8, 0.9], ["100%", "95%"]),
    bgHeight: useTransform(scrollYProgress, [0.7, 1], ["100%", "90%"]),

    // Add this new animation for text color transition
    gradientTextOpacity: useTransform(
      scrollYProgress,
      [0, 0.3],
      [0, 1]
    ),
    originalTextOpacity: useTransform(
      scrollYProgress,
      [0, 0.3],
      [1, 0]
    ),
    // Add sparkles opacity animation
    sparklesOpacity: useTransform(
      scrollYProgress,
      [0, 0.15, 0.3],
      [0, 1, 0.7]
    )
  }


  useEffect(() => {
    const unsubscribe = scrollYProgress.on("change", (latest) => {
      setIsFlipped(latest > 0.1);
      // Update our threshold state based on scroll position
      setIsPastThreshold(latest >= 0.3);
    });

    return () => unsubscribe();
  }, [scrollYProgress]);

  // For handling manual flip when not scrolling
  const handleFlip = () => {
    setIsFlipped(!isFlipped);
  };


  return (
    <motion.div ref={sectionRef} className="relative w-full h-[200vh] " initial={{ opacity: 0, y: -15 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: -15 }}
    >
      {/* This is the container that stays fixed in the viewport */}
      <div className="sticky top-0 h-screen w-full overflow-hidden flex justify-center">
        {/* Background */}
        <motion.div
          style={{ opacity: transformAnimations.bgOpacity, width: transformAnimations.bgWidth, height: transformAnimations.bgHeight }}
          className=" inset-0 h-[70%] rounded-xl transition-all duration-300 shadow-xl shadow-[#97E2F9] "
        />

        {/* Content Container */}
        <div className="absolute inset-0 w-full h-full flex items-center justify-center">
          {/* Text Content - positioned absolute */}
          <motion.div
            className="absolute top-[20%] w-full max-w-lg transition-all duration-300"
          >
            {/* The title with sparkles effect */}
            <motion.div
              className="relative mb-4"
              style={{
                transform: transformAnimations.h1Transform,
                fontSize: transformAnimations.titleFontSize
              }}
            >
              {/* Sparkles container */}
              <motion.div
                className="absolute w-full h-[22rem] top-[4.5rem] z-0"
                style={{ opacity: transformAnimations.sparklesOpacity }}
              >
                {/* Gradients */}
                <div className="absolute inset-x-10 top-0 bg-gradient-to-r from-transparent via-blue-500 to-transparent h-[2px] w-3/4 blur-sm" />
                <div className="absolute inset-x-10 top-0 bg-gradient-to-r from-transparent via-blue-500 to-transparent h-px w-3/4" />
                <div className="absolute inset-x-40 top-0 bg-gradient-to-r from-transparent via-teal-400 to-transparent h-[3px] w-1/3 blur-sm" />
                <div className="absolute inset-x-40 top-0 bg-gradient-to-r from-transparent via-teal-400 to-transparent h-px w-1/3" />

                {/* Core sparkles component */}
                <SparklesCore
                  background="transparent"
                  minSize={0.5}
                  maxSize={0.9}
                  particleDensity={1600}
                  className="w-full h-full"
                  particleColor="#0284c7"
                />

                {/* Radial gradient to fade edges */}
                <div className="absolute inset-0 w-full h-full [mask-image:radial-gradient(250px_60px_at_center,white_30%,transparent_80%)]"></div>
              </motion.div>

              {/* Actual title text */}
              <motion.h1
                className="relative font-bold text-gray-800 z-10"
              >
                Learning Languages{" "}

                {/* Use the state variable for conditional rendering */}
                {!isPastThreshold ? (
                  <span>Faster</span>
                ) : (
                  <div className="translate-x-[-3%]">
                    <ContainerTextFlip
                      animationDuration={700}
                      interval={2000}
                      words={["Faster", "Better", "Easier", "Elevated"]}
                    />
                  </div>
                )}
              </motion.h1>
            </motion.div>

            {/* Original h2 that fades out */}
            <motion.h2
              className="font-bold text-gray-500 mb-6 absolute w-full"
              style={{
                fontSize: transformAnimations.subtitleFontSize,
                transform: transformAnimations.h2Transform,
                opacity: transformAnimations.originalTextOpacity
              }}
            >
              Using AI Mnemonics
            </motion.h2>

            {/* Gradient h2 that fades in */}
            <motion.h2
              className="font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-teal-400 mb-6 absolute w-full"
              style={{
                fontSize: transformAnimations.subtitleFontSize,
                transform: transformAnimations.h2Transform,
                opacity: transformAnimations.gradientTextOpacity
              }}
            >
              Using AI Mnemonics
            </motion.h2>

            {/* Spacing element to maintain layout (same height as h2) */}
            <div className="mb-6" style={{ height: "1.5rem" }}></div>

            <motion.div style={{ opacity: transformAnimations.contentOpacity, transform: transformAnimations.pTransform }}>
              <p className="text-gray-600 mb-8 text-lg w-[70%]">
                Create personalized flashcards with powerful memory techniques that make
                learning engaging and effective.
              </p>
              <div className="flex flex-wrap gap-4">
                <Button text="Get Started" onClick={() => { }} />
                <Button text="See Demo" onClick={() => { }} />
              </div>
            </motion.div>
          </motion.div>

          {/* Flashcard container - positioned absolute */}
          <motion.div
            className="absolute left-1/2 top-[15%] transition-all duration-300"
            style={{ transform: transformAnimations.cardTransform }}
          >
            {/* Replaced the custom FlashCard with the imported Flashcard component */}
            <Flashcard
              card={cardData}
              isLoading={false}
              disableEdit={true}
              showFront={!isFlipped}
            />
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};

export default HeroSection;
