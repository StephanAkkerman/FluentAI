"use client"
import React, { useRef, useEffect, useState, useLayoutEffect } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ThreeDMarquee } from "../ui/3d-marquee";

const WhySection = () => {
    const sectionRef = useRef(null);
    const titleRef = useRef(null);

    // Add a state to control when to show content after title animation
    const [shouldShowContent, setShouldShowContent] = useState(false);

    // Add responsive state
    const [windowSize, setWindowSize] = useState<{ width: number; height: number }>(() => ({
        width: window.innerWidth,
        height: window.innerHeight
    }));

    // Update window size on resize
    useLayoutEffect(() => {
        function handleResize() {
            setWindowSize({ width: window.innerWidth, height: window.innerHeight });
        }
        window.addEventListener("resize", handleResize);
        handleResize();
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    // Screen size breakpoints
    const isSmallScreen = windowSize && windowSize.width < 768;

    const { scrollYProgress } = useScroll({
        target: sectionRef,
        offset: ["start start", "end end"]
    });

    // Title animation transformations
    const titleScale = useTransform(
        scrollYProgress,
        [0.2, 0.4],
        [1.5, 1]
    );
    // Title animation transformations
    const titleOpacity = useTransform(
        scrollYProgress,
        [0, 0.2],
        [0, 1]
    );

    const titleY = useTransform(
        scrollYProgress,
        [0.2, 0.4],
        isSmallScreen ?
            ["300%", "50%"] : ["300%", "0%"]
    );

    // Monitor scrollYProgress to determine when to show content
    useEffect(() => {
        const unsubscribe = scrollYProgress.onChange(value => {
            setShouldShowContent(value >= 0.4);
        });

        return () => unsubscribe();
    }, [scrollYProgress]);

    // Sample images array
    const images = [
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/duck.jpg",
        "/logo.png",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
        "/duck.jpg",
        "/logo.png",
    ];

    const fadeInLeft = {
        hidden: { opacity: 0, x: -30 },
        visible: { opacity: 1, x: 0 }
    };

    const fadeInRight = {
        hidden: { opacity: 0, x: 30 },
        visible: { opacity: 1, x: 0 }
    };

    const fadeInUp = {
        hidden: { opacity: 0, y: 30 },
        visible: { opacity: 1, y: 0 }
    };


    return (
        <section
            ref={sectionRef}
            className="relative w-full h-[300vh] pt-[8rem] md:pt-30"
        >
            {/* Main sticky container - adjusted for 80px header */}
            <div className="sticky top-[150px] left-0 w-full min-h-[120vh] sm:min-h-[100vh] h-screen flex flex-col overflow-hidden">
                {/* Background layer */}
                <div className="absolute inset-0 w-full h-full">
                    {/* Background overlay */}
                    <div className="absolute inset-0 w-full h-full z-10 bg-black/80 " />

                    {/* 3D Marquee Background */}
                    <ThreeDMarquee
                        className="pointer-events-none absolute inset-0 w-full h-full overflow-hidden"
                        images={images}
                    />
                </div>

                {/* Content */}
                <div className="relative z-20 h-full w-full px-4 container mx-auto flex flex-col justify-center">
                    {/* Title with animation */}
                    <motion.div
                        ref={titleRef}
                        className="flex flex-col items-center justify-center text-center md:mb-8"
                        style={{
                            scale: titleScale,
                            opacity: titleOpacity,
                            y: titleY,
                        }}
                    >
                        <h2 className="text-xl sm:text-4xl md:text-5xl font-bold text-white mb-2" >
                            Why Choose Our Approach
                        </h2>
                        <div className="h-1 w-24 bg-gradient-to-r from-blue-500 to-teal-400 rounded-full md:my-4"></div>
                    </motion.div>

                    {/* Comparison Content */}
                    <motion.div
                        className="w-full max-w-7xl mx-auto translate-y-[5%]"
                    >
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-16">
                            {/* Traditional Learning - Left Column */}
                            <motion.div
                                className="p-2 px-4 md:p-6 rounded-xl border border-gray-700 bg-black/30 h-full md:h-full"
                                variants={fadeInLeft}
                                initial="hidden"
                                animate={shouldShowContent ? "visible" : "hidden"}
                                transition={{ duration: 0.6 }}
                            >
                                <div className="flex flex-row md:flex-col justify-start items-center md:items-start">
                                    <div className="text-4xl md:mb-4">📚 </div>
                                    <h3 className="text-xl md:text-2xl font-bold text-white ml-4 md:ml-0 mb-0  border-gray-700 pb-1">
                                        Traditional Learning
                                    </h3>
                                </div>
                                <hr className="my-2 border-gray-700" />

                                <ul className="space-y-1 md:space-y-4">
                                    <motion.li
                                        className="flex items-start gap-3 text-gray-300"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.1 }}
                                    >
                                        <span className="text-red-400 mt-1">✗</span>
                                        <p>Repetitive memorization with minimal context</p>
                                    </motion.li>

                                    <motion.li
                                        className="flex items-start gap-3 text-gray-300"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.2 }}
                                    >
                                        <span className="text-red-400 mt-1">✗</span>
                                        <p>Generic learning materials not tailored to your preferences</p>
                                    </motion.li>

                                    <motion.li
                                        className="flex items-start gap-3 text-gray-300"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.3 }}
                                    >
                                        <span className="text-red-400 mt-1">✗</span>
                                        <p>Quick forgetting due to lack of meaningful associations</p>
                                    </motion.li>

                                    <motion.li
                                        className="flex items-start gap-3 text-gray-300"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.4 }}
                                    >
                                        <span className="text-red-400 mt-1">✗</span>
                                        <p>Boring, tedious process leading to low motivation</p>
                                    </motion.li>
                                </ul>
                            </motion.div>

                            {/* Our Approach - Right Column */}
                            <motion.div
                                className="p-2 px-4 md:p-6 rounded-xl border border-blue-500/30 bg-gradient-to-br from-blue-900/20 to-teal-900/20"
                                variants={fadeInRight}
                                initial="hidden"
                                animate={shouldShowContent ? "visible" : "hidden"}
                                transition={{ duration: 0.6 }}
                            >
                                <div className="flex flex-row md:flex-col justify-start items-center md:items-start">
                                    <div className="text-4xl md:mb-4">🧠</div>
                                    <h3 className="text-xl md:text-2xl font-bold text-white ml-4 md:ml-0 mb-0 pb-1">
                                        Our AI Approach
                                    </h3>

                                </div>
                                <hr className="my-2 border-blue-500/30" />

                                <ul className="space-y-1 md:space-y-4">
                                    <motion.li
                                        className="flex items-start gap-3 text-gray-200"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.1 }}
                                    >
                                        <span className="text-teal-400 mt-1">✓</span>
                                        <p>Personalized mnemonics crafted for your learning style</p>
                                    </motion.li>

                                    <motion.li
                                        className="flex items-start gap-3 text-gray-200"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.2 }}
                                    >
                                        <span className="text-teal-400 mt-1">✓</span>
                                        <p>Visual associations that connect to your existing knowledge</p>
                                    </motion.li>

                                    <motion.li
                                        className="flex items-start gap-3 text-gray-200"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.3 }}
                                    >
                                        <span className="text-teal-400 mt-1">✓</span>
                                        <p>Spaced repetition algorithm based on your retention patterns</p>
                                    </motion.li>

                                    <motion.li
                                        className="flex items-start gap-3 text-gray-200"
                                        variants={fadeInUp}
                                        transition={{ delay: 0.4 }}
                                    >
                                        <span className="text-teal-400 mt-1">✓</span>
                                        <p>Engaging, game-like experience that keeps you motivated</p>
                                    </motion.li>
                                </ul>
                            </motion.div>
                        </div>

                        {/* Bottom CTA */}
                        <motion.div
                            className="mt-4 md:mt-12 text-center"
                            variants={fadeInUp}
                            initial="hidden"
                            animate={shouldShowContent ? "visible" : "hidden"}
                            transition={{ duration: 0.6, delay: 0.5 }}
                        >
                            <button className="px-8 py-3 bg-gradient-to-r from-blue-500 to-teal-400 text-white font-medium rounded-full hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-300">
                                Start Learning Smarter
                            </button>
                        </motion.div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
};

export default WhySection;