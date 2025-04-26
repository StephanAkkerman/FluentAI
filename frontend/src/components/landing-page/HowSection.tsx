

"use client";

import React, { useRef, useEffect, useState, ReactNode } from "react";
import { motion, useScroll, useTransform, MotionValue } from "framer-motion";
import Browser from "../ui/Browser";
import Flashcard from "../Flashcard";
import flashe from "../../../public/Flashy_flashe.png"


const steps = [
  {
    threshold: 0.22,
    content: (
      <div className="w-full aspect-video overflow-hidden">
        <video
          src="/step1.mp4"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />
      </div>
    ),
  },
  {
    threshold: 0.5,
    content: (
      <div className="w-full aspect-video overflow-hidden">
        <video
          src="/step2.mp4"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />
      </div>
    ),
  },
  {
    threshold: Infinity, // applies to any value ≥ 0.5
    content: (
      <div className="w-full aspect-video overflow-hidden">
        <video
          src="/step3.mp4"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />
      </div>
    ),
  },
];

const sections = [
  {
    transformRange: [0, 0.2],
    transformValues: ["translate(55%, 0%)", "translate(55%, -600%)"],
    opacityRange: [0, 0.03, 0.15, 0.2],
    opacityValues: ["0", "1", "1", "0"],
    title: "Select Your Target Language",
    description:
      "Choose the language you want to learn from our growing collection of supported languages, with the word you want to learn",
  },
  {
    transformRange: [0.2, 0.5],
    transformValues: ["translate(0%, 0%)", "translate(0%, -775%)"],
    opacityRange: [0.23, 0.26, 0.5, 0.55],
    opacityValues: ["0", "1", "1", "0"],
    title: "Build Your Learning Path",
    description:
      "Either select specific words you want to master or let Mnemorai suggest an optimized learning route based on frequency and usefulness. You control what you learn and when you learn it.",
  },
  {
    transformRange: [0.45, 0.75],
    transformValues: ["translate(60%, 0%)", "translate(60%, -775%)"],
    opacityRange: [0.5, 0.51, 0.8, 0.85],
    opacityValues: ["0", "1", "1", "0"],
    title: "Discover Memory-Boosting Flashcards",
    description:
      "Mnemorai automatically generates personalized flashcards for each word, featuring clever mnemonic phrases and vivid images that connect the foreign word to familiar sounds and concepts.",
  },
];


type SectionBlockProps = {
  data: typeof sections[number];
  scrollYProgress: MotionValue<number>;
};

const SectionBlock: React.FC<SectionBlockProps> = ({ data, scrollYProgress }) => {
  const transform = useTransform(
    scrollYProgress,
    data.transformRange,
    data.transformValues
  );
  const opacity = useTransform(
    scrollYProgress,
    data.opacityRange,
    data.opacityValues
  );

  return (
    <motion.div
      style={{ transform, opacity }}
      className="transition-all duration-300"
    >
      <div className="bg-gradient-to-r from-blue-500 to-teal-400 rounded-xl w-[40%] p-1">
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4">
          <h3 className="font-bold text-2xl mb-4">{data.title}</h3>
          <p className="text-base">{data.description}</p>
        </div>
      </div>
    </motion.div>
  );
};


const HowSection: React.FC = () => {
  /* ---------- Scroll tracking ---------- */
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: sectionRef });

  /* ---------- Step-video handling ------- */
  const [browserContent, setBrowserContent] = useState<ReactNode>(
    steps[0].content
  );

  useEffect(
    () =>
      scrollYProgress.onChange((v) => {
        const step = steps.find((s) => v < s.threshold)!; // safe; last threshold = Infinity
        setBrowserContent(step.content);
      }),
    [scrollYProgress]
  );


  const browserTransform = useTransform(
    scrollYProgress,
    [0.22, 0.225, 0.5, 0.505],
    [
      "translate(0%, 0%)",
      "translate(50%, 0%)",
      "translate(50%, 0%)",
      "translate(0%, 0%)",
    ]
  );
  const browserOpacity = useTransform(scrollYProgress, [0.65, 0.655], ["1", "0"]);


  const [showCard, setShowCard] = useState(false);
  useEffect(
    () => scrollYProgress.onChange((v) => setShowCard(v >= 0.655)),
    [scrollYProgress]
  );

  const cardOpacity = useTransform(scrollYProgress, [0.66, 0.67], ["0", "1"]);
  const cardTransform = useTransform(
    scrollYProgress,
    [0.66, 0.67],
    ["translate(-2%, -10%)", "translate(-2%, 0%)"]
  );

  const cardData = {
    word: "Flashe",
    imageUrl: flashe.src,
    audioUrl: "",
    ipa: "flaʃə",
    verbalCue: "Imagine a bottle so flashy, it's like it's got its own built-in disco ball.",
    translation: "Bottle",
    languageCode: "de",
  };


  return (
    <section
      ref={sectionRef}
      id="how-it-works"
      className="relative w-full h-[650vh]"
    >
      <div className="container mx-auto px-6 sticky top-[8rem] overflow-hidden">
        <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-white mb-12">
          How It Works
        </h2>

        {/* Browser mock-up */}
        <motion.div
          style={{ transform: browserTransform, opacity: browserOpacity }}
          className="transition-all duration-300"
        >
          <div className="relative w-full h-full max-w-[500px] max-h-[750px]">
            <Browser urlText="https://mnemorai.com" dark className="cursor-pointer">
              {browserContent}
            </Browser>
          </div>
        </motion.div>

        {/* Explanatory blocks */}
        {sections.map((sec, idx) => (
          <SectionBlock key={idx} data={sec} scrollYProgress={scrollYProgress} />
        ))}

        {/* Flashcard reveal */}
        <motion.div
          className="absolute w-full flex flex-col top-20 px-6 mx-auto items-center justify-center transition-all duration-300"
          style={{
            opacity: cardOpacity,
            transform: cardTransform,
            pointerEvents: showCard ? "auto" : "none",
          }}
        >
          <div className="text-center mb-20">
            <h3 className="font-bold text-2xl mb-4 gradient-text">Et voilà!</h3>
            <p className="text-base">
              Just like that you created a{" "}
              <span className="gradient-text">personalized</span>
              <br />
              mnemonic-empowered flashcard!
            </p>
          </div>
          <Flashcard isLoading={false} card={cardData} />
        </motion.div>
      </div>
    </section>
  );
};

export default HowSection;
