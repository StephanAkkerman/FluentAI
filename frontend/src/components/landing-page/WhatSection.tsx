import React, { useRef, useState } from "react";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { motion, useInView } from "motion/react";
import { TextGenerateEffect } from "../ui/text-generate-effect";
import Flashcard from "../Flashcard";
import { Card } from "@/interfaces/CardInterfaces";
import LanguageDock from "../ui/language-dock";

import duck from "../../../public/duck.jpg";

const WhatSection = () => {
    const sectionRef = useRef(null);
    const isInView = useInView(sectionRef, { once: true, amount: 0.1 });

    const features = [
        {
            title: "Track issues effectively",
            description:
                "Track and manage your project issues with ease using our intuitive interface.",
            skeleton: <SkeletonOne />,
            className:
                "col-span-1 lg:col-span-3 border-b lg:border-r dark:border-neutral-800",
        },
        {
            title: "Learn quicker through flashcards",
            description:
                "Create personalized flashcards effortlessly using our advanced AI technology.",
            skeleton: <SkeletonTwo />,
            className: "border-b col-span-1 lg:col-span-3 dark:border-neutral-800",
        },
        {
            title: "Current supported languages",
            description:
                "Our team is constantly working hard to include more languages for you to learn. So stay tuned!",
            skeleton: <SkeletonThree />,
            className:
                "col-span-1 lg:col-span-3 lg:border-r dark:border-neutral-800",
        },
        {
            title: "Deploy in seconds",
            description:
                "With our blazing fast, state of the art, cutting edge, we are so back cloud servies (read AWS) - you can deploy your model in seconds.",
            skeleton: <SkeletonFour />,
            className: "col-span-1 lg:col-span-3 border-b lg:border-none",
        },
    ];

    return (
        <div className="relative z-20 py-20 max-w-7xl mx-auto">
            <motion.div
                className="px-8"
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                transition={{ duration: 0.6 }}
            >
                <h1 className="text-3xl lg:text-5xl lg:leading-tight max-w-5xl mx-auto text-center tracking-tight font-medium text-black dark:text-white">
                    <TextGenerateEffect words={'Learning languages has never been easier'} className="text-3xl" />
                </h1>

                <motion.p
                    className="text-sm lg:text-base max-w-2xl my-4 mx-auto text-neutral-500 text-center font-normal dark:text-neutral-300"
                    initial={{ opacity: 0 }}
                    animate={isInView ? { opacity: 1 } : { opacity: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 }}
                >
                    From Image generation to video generation, Everything AI has APIs for
                    literally everything. It can even create this website copy for you.
                </motion.p>
            </motion.div>

            <div className="relative" ref={sectionRef} >
                <div className="grid grid-cols-1 lg:grid-cols-6 mt-12 xl:border rounded-md dark:border-neutral-800">
                    {features.map((feature, index) => (
                        <AnimatedFeatureCard
                            key={feature.title}
                            className={feature.className}
                            index={index}
                            isInView={isInView}
                        >
                            <FeatureTitle>{feature.title}</FeatureTitle>
                            <FeatureDescription>{feature.description}</FeatureDescription>
                            <div className="h-[90%] w-full">{feature.skeleton}</div>
                        </AnimatedFeatureCard>
                    ))}
                </div>
            </div>
        </div>
    );
};

const AnimatedFeatureCard = ({
    children,
    className,
    index,
    isInView,
}: {
    children?: React.ReactNode;
    className?: string;
    index: number;
    isInView: boolean;
}) => {
    return (
        <motion.div
            className={cn(`p-4 sm:p-8 relative overflow-hidden h-[90%]`, className)}
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
            transition={{
                duration: 0.5,
                delay: 0.5 + (index * 0.1),  // Staggered animation
            }}
        >
            {children}
        </motion.div>
    );
};

const FeatureTitle = ({ children }: { children?: React.ReactNode }) => {
    return (
        <p className="max-w-5xl mx-auto text-left tracking-tight text-black dark:text-white text-xl md:text-2xl md:leading-snug">
            {children}
        </p>
    );
};

const FeatureDescription = ({ children }: { children?: React.ReactNode }) => {
    return (
        <p
            className={cn(
                "text-sm md:text-base max-w-4xl text-left mx-auto",
                "text-neutral-500 text-center font-normal dark:text-neutral-300",
                "text-left max-w-sm mx-0 md:text-sm my-2"
            )}
        >
            {children}
        </p>
    );
};

export const SkeletonOne = () => {
    const [isHovered, setIsHovered] = useState(false);

    return (
        <div className="relative flex pt-8 w-full items-start justify-center gap-10 h-[80%]">
            <div
                className="flex h-full w-[60%] items-start justify-center"
                style={{
                    transformStyle: "preserve-3d",
                    transform: "rotateY(-15deg) rotateX(5deg)",
                    perspective: "1000px"
                }}
            >
                <motion.div
                    initial={{
                        transform: "translateZ(5px) translateY(0px)",
                    }}
                    animate={isHovered ? {
                        transform: "translateZ(20px) translateY(-10px)",
                    } : {
                        transform: "translateZ(5px) translateY(0px)",
                    }}
                    transition={{
                        duration: 0.5,
                        ease: "easeInOut",
                    }}
                    className="rounded-xl overflow-hidden border border-neutral-200 dark:border-neutral-700 transition-all duration-300"
                    onMouseEnter={() => setIsHovered(true)}
                    onMouseLeave={() => setIsHovered(false)}
                    style={{
                        boxShadow: isHovered
                            ? "0 20px 30px rgba(59, 131, 246, 0.3), 0 20px 30px rgba(45, 212, 190, 0.3)"
                            : "0 5px 15px rgba(0, 0, 0, 0.1)"
                    }}
                >
                    <Image
                        src="/infographic.png"
                        alt="infographic"
                        width={700}
                        height={400}
                        className="object-cover rounded-xl"
                    />
                </motion.div>
            </div>
        </div>
    );
};

export const SkeletonTwo = () => {
    // Create multiple card data objects for different words
    const cardData = [
        {
            word: "Pato",
            imageUrl: duck.src,
            audioUrl: "",
            ipa: "/ˈpɑtoʊ/",
            verbalCue: "Spanish word for 'duck'",
            translation: "Duck",
            languageCode: "es"
        },
        {
            word: "Gato",
            imageUrl: "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
            audioUrl: "",
            ipa: "/ˈɡɑtoʊ/",
            verbalCue: "Spanish word for 'cat'",
            translation: "Cat",
            languageCode: "es"
        },
        {
            word: "Perro",
            imageUrl: "https://images.unsplash.com/photo-1517849845537-4d257902454a",
            audioUrl: "",
            ipa: "/ˈpɛroʊ/",
            verbalCue: "Spanish word for 'dog'",
            translation: "Dog",
            languageCode: "es"
        },
        {
            word: "Casa",
            imageUrl: "https://images.unsplash.com/photo-1518780664697-55e3ad937233",
            audioUrl: "",
            ipa: "/ˈkɑsɑ/",
            verbalCue: "Spanish word for 'house'",
            translation: "House",
            languageCode: "es"
        },
        {
            word: "Libro",
            imageUrl: "https://images.unsplash.com/photo-1544947950-fa07a98d237f",
            audioUrl: "",
            ipa: "/ˈlibɾo/",
            verbalCue: "Spanish word for 'book'",
            translation: "Book",
            languageCode: "es"
        },
    ];

    const imageVariants = {
        whileHover: {
            scale: 1.1,
            rotate: 0,
            zIndex: 100,
        },
        whileTap: {
            scale: 1.1,
            rotate: 0,
            zIndex: 100,
        },
    };

    const FlashcardWrapper = ({ card, index }: { card: Card; index: number }) => (
        <motion.div
            variants={imageVariants}
            style={{
                rotate: Math.random() * 20 - 10,
            }}
            whileHover="whileHover"
            whileTap="whileTap"
            className="rounded-xl  mt-4  dark:bg-neutral-800 dark:border-neutral-700 border border-neutral-100 shrink-0 overflow-hidden"
        >
            {/* This wrapper preserves the aspect ratio while scaling down */}
            <div
                className="relative"
                style={{
                    width: '120px',
                    height: '144px', // Maintains 5:6 aspect ratio (80:96 scaled down)
                    overflow: 'hidden'
                }}
            >
                <div
                    className="absolute top-0 left-0 transform origin-top-left"
                    style={{
                        transform: 'scale(0.375)',
                        width: '320px',
                        height: '384px'
                    }}
                >
                    <Flashcard
                        card={card}
                        isLoading={false}
                        showFront={index % 2 === 0}
                    />
                </div>
            </div>
        </motion.div>
    );

    return (
        <div className="relative flex flex-col items-start p-8 pt-10 gap-5 h-full overflow-hidden rounded-xl">
            {/* First row of flashcards */}
            <div className="flex flex-row -ml-20">
                {cardData.map((card, idx) => (
                    <FlashcardWrapper
                        key={"flashcard-first" + idx}
                        card={card}
                        index={idx}
                    />
                ))}
            </div>

            {/* Second row with more flashcards */}
            <div className="flex flex-row">
                {cardData.slice().reverse().map((card, idx) => (
                    <FlashcardWrapper
                        key={"flashcard-second" + idx}
                        card={card}
                        index={idx + cardData.length} // Different index to alternate front/back display
                    />
                ))}
            </div>

            {/* Third row of flashcards */}
            <div className="flex flex-row -ml-20">
                {cardData.map((card, idx) => (
                    <FlashcardWrapper
                        key={"flashcard-third" + idx}
                        card={card}
                        index={idx}
                    />
                ))}
            </div>

            <div className="absolute left-0 z-[100] inset-y-0 w-20 bg-gradient-to-r from-white dark:from-black to-transparent h-full pointer-events-none" />
            <div className="absolute right-0 z-[100] inset-y-0 w-20 bg-gradient-to-l from-white dark:from-black to-transparent h-full pointer-events-none" />
        </div>
    );
};

export const SkeletonThree = () => {
    return (
        <>
            {/* 4. LANGUAGES SECTION */}
            <div className="container mx-auto px-6">
                <div className="flex flex-wrap justify-center gap-4 max-w-4xl mt-20 mx-auto">
                    <LanguageDock />
                </div>
            </div>
        </>
    );
};

export const SkeletonFour = () => {
    return (
        <div className="h-60 md:h-60 flex flex-col items-center relative bg-transparent dark:bg-transparent mt-10">
        </div>
    );
};

export default WhatSection;