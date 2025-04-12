"use client"
import React, { useRef, useEffect, useState } from "react";
import { motion, useScroll, useTransform, Variants } from "framer-motion";

// Import the ColourfulText component
import { ColourfulText } from "../ui/colorful-text";

interface AIFeature {
    icon: string;
    title: string;
    description: string;
}

const AIPage: React.FC = () => {
    const sectionRef = useRef<HTMLElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    const [shouldShowContent, setShouldShowContent] = useState(false);

    const { scrollYProgress } = useScroll({
        target: sectionRef,
        offset: ["start end", "end start"]
    });

    // Title animation transformations
    const titleScale = useTransform(
        scrollYProgress,
        [0.2, 0.3, 0.4],
        [1.2, 1, 1]
    );
    const titleOpacity = useTransform(
        scrollYProgress,
        [0.2, 0.3, 0.4],
        [0, 1, 1]
    );
    const titleY = useTransform(
        scrollYProgress,
        [0.2, 0.3, 0.4, 0.5],
        ["700%", "500%", "500%", "40%"]
    );

    // Features animation transformations based on scroll
    const featuresOpacity = useTransform(
        scrollYProgress,
        [0.5, 0.65],  // Start at 50%, fully visible at 65%
        [0, 1]
    );

    // Control video playback based on scroll
    useEffect(() => {
        const updateVideoTime = (value: number) => {
            if (!videoRef.current) return;

            // First time setting video properties
            if (videoRef.current.duration && !isNaN(videoRef.current.duration)) {
                // Set the current time based on scroll progress
                const newTime = value * videoRef.current.duration;

                // Only update if the change is significant to avoid performance issues
                if (Math.abs(videoRef.current.currentTime - newTime) > 0.5) {
                    videoRef.current.currentTime = newTime;
                }
            }

            // Set the content visibility state based on scroll position
            // Show content when we're halfway through the section
            setShouldShowContent(value >= 0.55);
        };

        const unsubscribe = scrollYProgress.onChange(updateVideoTime);
        return () => unsubscribe();
    }, [scrollYProgress]);

    // Overlay animation based on scroll
    const overlayOpacity = useTransform(
        scrollYProgress,
        [0.3, 0.5, 0.7],
        [0.1, 0.3, 0.5] // Start lighter, then get darker as user scrolls
    );

    // Animation variants for content blocks
    const fadeInUp = {
        hidden: { opacity: 0, y: 30 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.8 } }
    };

    // Animation variants for the feature items with staggered animation
    const fadeInLeft = {
        hidden: { opacity: 0, x: -30 },
        visible: { opacity: 1, x: 0, transition: { duration: 0.4 } }

    };

    const fadeInRight = {
        hidden: { opacity: 0, x: 30 },
        visible: { opacity: 1, x: 0, transition: { duration: 0.4 } }

    };

    // AI-powered language learning features
    const aiFeatures: AIFeature[] = [
        {
            icon: "🔍",
            title: "Smart Mnemonic Generation",
            description: "Our AI analyzes words to create memory-boosting associations tailored to how your brain learns"
        },
        {
            icon: "🔊",
            title: "Phonetic Matching",
            description: "Discover similar-sounding words that bridge your native language with new vocabulary"
        },
        {
            icon: "🖼️",
            title: "Visual Learning Anchors",
            description: "Custom images that connect new words to concepts you already understand"
        },
        {
            icon: "🧠",
            title: "Personalized Learning Path",
            description: "AI adapts to your progress, focusing on what you need most"
        }
    ];

    return (
        <section
            ref={sectionRef}
            className="relative w-full h-[320vh]" // Taller to allow for more scroll-triggered animations
        >
            {/* Sticky container */}
            <div className="sticky top-0 left-0 w-full h-screen flex items-center justify-center overflow-hidden">
                {/* Video Background */}
                <div className="absolute inset-0 w-full h-full">
                    <video
                        ref={videoRef}
                        className="absolute inset-0 w-full h-full object-cover"
                        preload="auto"
                        muted
                        playsInline
                    >
                        <source src="/tech.mp4" type="video/mp4" />
                        {/* Fallback to static image if video fails */}
                        <img
                            src="/tech.mp4"
                            alt="AI technology"
                            className="absolute inset-0 w-full h-full object-cover"
                        />
                    </video>

                    {/* Overlay with dynamic opacity */}
                    <motion.div
                        className="absolute inset-0 w-full h-full bg-black backdrop-blur-sm"
                        style={{ opacity: overlayOpacity }}
                    />
                </div>

                {/* Content Container */}
                <div className="relative z-10 w-full max-w-6xl mx-auto px-4">
                    {/* Header - Always visible but animated in */}
                    <motion.div
                        className="text-center mb-12"
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                        style={{
                            scale: titleScale,
                            opacity: titleOpacity,
                            y: titleY,
                        }}
                    >
                        <h2 className="text-4xl md:text-5xl font-bold mb-4 text-white text-shadow-lg">
                            Language Learning
                            <ColourfulText text=" Powered by AI" />

                        </h2>
                        <div className="h-1 w-24 bg-gradient-to-r from-blue-500 to-teal-400 rounded-full mx-auto"></div>
                    </motion.div>

                    {/* Main Content - Only appears when scrolled to halfway point */}
                    <motion.div
                        className="w-full"
                        initial="hidden"
                        animate={shouldShowContent ? "visible" : "hidden"}
                    >
                        {/* How It Works - Brief Overview */}
                        <motion.div
                            className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 mb-4 max-w-3xl mx-auto"
                            variants={fadeInUp}
                        >
                            <h3 className="text-2xl font-semibold text-white mb-4">How Mnemorai Works</h3>
                            <p className="text-gray-200 mb-4">
                                Our AI system builds powerful memory connections by finding words in your native language that sound similar to new vocabulary. These phonetic bridges, combined with vivid imagery, create lasting memory anchors that make recall effortless.
                            </p>
                            <p className="text-gray-200">
                                Unlike traditional flashcards that rely on repetition alone, Mnemorai creates meaningful associations that work with your brain's natural ability to remember stories and images.
                            </p>
                        </motion.div>

                        {/* AI Features Grid - Now using the scrollYProgress for smooth animation */}
                        <motion.div
                            className="grid grid-cols-1 md:grid-cols-2 gap-8"
                            style={{ opacity: featuresOpacity }}
                        >
                            {aiFeatures.map((feature, index) => (
                                <motion.div
                                    key={index}
                                    className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-300"
                                    variants={index % 2 === 0 ? fadeInLeft : fadeInRight}
                                    custom={index}
                                >
                                    <div className="text-3xl mb-4">{feature.icon}</div>
                                    <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                                    <p className="text-gray-300">{feature.description}</p>
                                </motion.div>
                            ))}
                        </motion.div>

                        {/* CTA Button */}
                        <motion.div
                            className="mt-12 text-center"
                            variants={fadeInUp}
                        >
                            <button className="px-8 py-3 bg-gradient-to-r from-blue-500 to-teal-400 text-white font-medium rounded-full hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-300">
                                Try Mnemorai Now
                            </button>
                        </motion.div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
};

export default AIPage;