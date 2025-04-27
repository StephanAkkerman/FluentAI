"use client"
import React, { useRef, useEffect, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ThreeDMarquee } from "../ui/3d-marquee";

const WhySection = () => {
  const sectionRef = useRef(null);
  const titleRef = useRef(null);
  const [shouldShowContent, setShouldShowContent] = useState(false);
  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const update = () => {
      if (typeof window !== "undefined") {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight });
      }
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const isSmallScreen = windowSize && windowSize.width < 768;

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"]
  });

  const titleScale = useTransform(scrollYProgress, [0.15, 0.3], [1.8, 1]);
  const titleOpacity = useTransform(scrollYProgress, [0, 0.15], [0, 1]);
  const titleY = useTransform(
    scrollYProgress,
    [0.15, 0.3],
    isSmallScreen ? ["250%", "50%"] : ["200%", "0%"]
  );

  // Animate glow effect
  const glowOpacity = useTransform(scrollYProgress, [0.3, 0.4], [0, 0.6]);
  const glowScale = useTransform(scrollYProgress, [0.3, 0.4], [0.8, 1.1]);

  useEffect(() => {
    const unsubscribe = scrollYProgress.onChange(value => {
      setShouldShowContent(value >= 0.3);
    });
    return () => unsubscribe();
  }, [scrollYProgress]);

  // Sample images array
  const images = [
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
    "/logo.png",
  ];

  const fadeInLeft = {
    hidden: { opacity: 0, x: -40 },
    visible: { opacity: 1, x: 0 }
  };

  const fadeInRight = {
    hidden: { opacity: 0, x: 40 },
    visible: { opacity: 1, x: 0 }
  };

  const fadeInUp = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0 }
  };

  const staggerItems = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-[250vh] pt-[8rem] md:pt-30"
    >
      {/* Main sticky container */}
      <div className="sticky top-[150px] left-0 w-full min-h-[120vh] sm:min-h-[100vh] h-screen flex flex-col overflow-hidden">
        {/* Background layer */}
        <div className="absolute inset-0 w-full h-full">
          {/* Background overlay - gradient */}
          <div className="absolute inset-0 w-full h-full z-10 bg-gradient-to-b from-black/95 via-black/90 to-black/80" />

          {/* 3D Marquee Background */}
          <ThreeDMarquee
            className="pointer-events-none absolute inset-0 w-full h-full overflow-hidden"
            images={images}
          />

          {/* Animated glow effect */}
          <motion.div
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-blue-500 blur-[120px] z-5"
            style={{
              opacity: glowOpacity,
              scale: glowScale
            }}
          />
        </div>

        {/* Content */}
        <div className="relative z-20 h-full w-full px-4 container mx-auto flex flex-col justify-center">
          {/* Title with animation */}
          <motion.div
            ref={titleRef}
            className="flex flex-col items-center justify-center text-center md:mb-12"
            style={{
              scale: titleScale,
              opacity: titleOpacity,
              y: titleY,
            }}
          >
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.8 }}
              className="mb-3 text-blue-400 font-medium tracking-wider text-sm md:text-base uppercase"
            >
              SUPERIOR LEARNING TECHNOLOGY
            </motion.div>

            <h2 className="text-2xl sm:text-5xl md:text-6xl font-extrabold text-white mb-2">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-400">
                Why Choose
              </span>{" "}
              Our Approach
            </h2>

            <div className="h-1 w-32 bg-gradient-to-r from-blue-500 to-teal-400 rounded-full my-6"></div>

            <p className="text-gray-300 max-w-2xl text-sm md:text-lg">
              Our AI-driven approach to learning creates lasting connections in your brain
              that make information retrieval natural and effortless
            </p>
          </motion.div>

          {/* Comparison Content */}
          <motion.div className="w-full max-w-7xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10">
              {/* Traditional Learning - Left Column */}
              <motion.div
                className="p-6 md:p-8 rounded-2xl border border-gray-800 bg-gradient-to-br from-gray-900/50 to-black backdrop-blur-md group hover:border-gray-700 transition-all duration-300"
                variants={fadeInLeft}
                initial="hidden"
                animate={shouldShowContent ? "visible" : "hidden"}
                transition={{ duration: 0.6 }}
                whileHover={{ y: -5, transition: { duration: 0.3 } }}
              >
                <div className="flex flex-col justify-start items-start mb-6">
                  <div className="flex items-center space-x-4">
                    <div className="text-4xl p-3 bg-gray-800/50 rounded-xl">📚</div>
                    <h3 className="text-2xl md:text-3xl font-bold text-white">
                      Traditional Learning
                    </h3>
                  </div>
                  <div className="w-full h-px bg-gradient-to-r from-gray-800 via-gray-700 to-gray-800 my-6"></div>
                </div>

                <motion.ul
                  className="space-y-5"
                  variants={staggerItems}
                >
                  {[
                    "Repetitive memorization with minimal context",
                    "Generic learning materials not tailored to your preferences",
                    "Quick forgetting due to lack of meaningful associations",
                    "Boring, tedious process leading to low motivation"
                  ].map((item, index) => (
                    <motion.li
                      key={index}
                      className="flex items-start gap-4 text-gray-300 group-hover:text-gray-200 transition-colors duration-300"
                      variants={fadeInUp}
                    >
                      <span className="text-red-400 mt-1 text-lg font-bold">✗</span>
                      <p className="text-base md:text-lg">{item}</p>
                    </motion.li>
                  ))}
                </motion.ul>
              </motion.div>

              {/* Our Approach - Right Column */}
              <motion.div
                className="p-6 md:p-8 rounded-2xl border border-blue-600/30 bg-gradient-to-br from-blue-900/20 to-blue-800/10 backdrop-blur-md relative overflow-hidden group hover:border-blue-500/50 transition-all duration-300"
                variants={fadeInRight}
                initial="hidden"
                animate={shouldShowContent ? "visible" : "hidden"}
                transition={{ duration: 0.6 }}
                whileHover={{ y: -5, transition: { duration: 0.3 } }}
              >
                {/* Background glow */}
                <div className="absolute top-0 -right-20 w-40 h-40 bg-blue-500/20 rounded-full blur-3xl group-hover:bg-blue-500/30 transition-all duration-500"></div>

                <div className="flex flex-col justify-start items-start mb-6 relative z-10">
                  <div className="flex items-center space-x-4">
                    <div className="text-4xl p-3 bg-blue-800/30 rounded-xl bg-gradient-to-br from-blue-700/30 to-blue-900/30">🧠</div>
                    <h3 className="text-2xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-400">
                      Our AI Approach
                    </h3>
                  </div>
                  <div className="w-full h-px bg-gradient-to-r from-blue-800/30 via-teal-500/30 to-blue-800/30 my-6"></div>
                </div>

                <motion.ul
                  className="space-y-5 relative z-10"
                  variants={staggerItems}
                >
                  {[
                    "Personalized mnemonics crafted for your learning style",
                    "Visual associations that connect to your existing knowledge",
                    "Spaced repetition algorithm based on your retention patterns",
                    "Engaging, game-like experience that keeps you motivated"
                  ].map((item, index) => (
                    <motion.li
                      key={index}
                      className="flex items-start gap-4 text-gray-200 group-hover:text-white transition-colors duration-300"
                      variants={fadeInUp}
                    >
                      <span className="text-teal-400 mt-1 text-lg font-bold">✓</span>
                      <p className="text-base md:text-lg">{item}</p>
                    </motion.li>
                  ))}
                </motion.ul>
              </motion.div>
            </div>

            {/* Bottom CTA */}
            <motion.div
              className="mt-10 md:mt-16 text-center"
              variants={fadeInUp}
              initial="hidden"
              animate={shouldShowContent ? "visible" : "hidden"}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              <button className="px-10 py-4 bg-gradient-to-r from-blue-600 to-teal-500 text-white font-semibold text-lg rounded-xl hover:shadow-lg hover:shadow-blue-500/30 hover:scale-105 transition-all duration-300 relative group overflow-hidden">
                <span className="relative z-10">Start Learning Smarter</span>
                <span className="absolute inset-0 bg-gradient-to-r from-blue-500 to-teal-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
              </button>

              <p className="text-gray-400 mt-4 text-sm">
                Join thousands of students already using our AI approach
              </p>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default WhySection;
