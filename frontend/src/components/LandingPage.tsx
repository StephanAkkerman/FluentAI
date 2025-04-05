"use client"
import LoadingScreen from "./LoadingScreen";
import WhatSection from "./landing-page/WhatSection"
import WhySection from "./landing-page/WhySection";
import HowSection from "./landing-page/HowSection";
import HeroSection from "./landing-page/HeroSection";
import React, { useEffect, useState, useRef } from "react";

import { LanguageDock } from "@/components/ui/language-dock";




const LandingPage = () => {
    // State for interactive elements like flashcard flip
    const [isFlipped, setIsFlipped] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // On mount, check localStorage to determine if the user has already seen the loading screen.
    useEffect(() => {
        if (sessionStorage.getItem('hasVisited')) {
            setIsLoading(false);
        } else {
            sessionStorage.setItem('hasVisited', 'true');
        }
    }, []);

    // Callback passed to LoadingScreen when its fade-out completes.
    const handleLoadingComplete = () => {
        setIsLoading(false);
    };

    // Toggle flashcard flip
    const handleFlip = () => {
        setIsFlipped(!isFlipped);
    };

    return (
        <>
            {isLoading && <LoadingScreen onComplete={handleLoadingComplete} />}


            <div className="min-h-screen flex flex-col w-full">
                <section id="home">
                    <HeroSection />
                </section>


                <section id="what" className="duration-300 transition-all -translate-y-[5%]">
                    <WhatSection />
                </section>

                <section id="why" className="duration-300 transition-all -translate-y-[5%]">
                    <WhySection />
                </section>

                {/* 2. HOW IT WORKS SECTION */}
                < section id="how" className="py-20" >
                    <HowSection />
                </section >

                {/* 3. FEATURES & BENEFITS SECTION */}
                < section id="features" className="py-20 bg-[#F6CBAF] bg-opacity-20" >
                    <div className="container mx-auto px-6">
                        <h2 className="text-3xl font-bold text-center text-gray-800 mb-12">
                            Features & Benefits
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
                            {/* Feature 1 */}
                            <div className="bg-white p-8 rounded-xl shadow-sm border border-[#97E2F9]">
                                <div className="w-16 h-16 rounded-full bg-[#97E2F9] bg-opacity-30 flex items-center justify-center text-2xl mb-6 mx-auto">
                                    🧠
                                </div>
                                <h3 className="text-xl font-bold text-center text-gray-800 mb-4">
                                    Personalized Mnemonics
                                </h3>
                                <p className="text-gray-600 text-center">
                                    AI creates unique memory techniques tailored to how your brain learns best.
                                </p>
                            </div>

                            {/* Feature 2 */}
                            <div className="bg-white p-8 rounded-xl shadow-sm border border-[#F6CBAF]">
                                <div className="w-16 h-16 rounded-full bg-[#F6CBAF] bg-opacity-30 flex items-center justify-center text-2xl mb-6 mx-auto">
                                    ⏱️
                                </div>
                                <h3 className="text-xl font-bold text-center text-gray-800 mb-4">
                                    Spaced Repetition
                                </h3>
                                <p className="text-gray-600 text-center">
                                    Smart algorithm schedules reviews at optimal times for maximum retention.
                                </p>
                            </div>

                            {/* Feature 3 */}
                            <div className="bg-white p-8 rounded-xl shadow-sm border border-[#26C485]">
                                <div className="w-16 h-16 rounded-full bg-[#26C485] bg-opacity-30 flex items-center justify-center text-2xl mb-6 mx-auto">
                                    📱
                                </div>
                                <h3 className="text-xl font-bold text-center text-gray-800 mb-4">
                                    Interactive Flashcards
                                </h3>
                                <p className="text-gray-600 text-center">
                                    Engaging flip animations with audio pronunciation and progress tracking.
                                </p>
                            </div>
                        </div>
                    </div>
                </section >



                {/* 5. FAQ SECTION */}
                < section id="faq" className="py-20 bg-[#97E2F9] bg-opacity-10" >
                    <div className="container mx-auto px-6">
                        <h2 className="text-3xl font-bold text-center text-gray-800 mb-12">
                            Frequently Asked Questions
                        </h2>
                        <div className="max-w-3xl mx-auto space-y-6">
                            {/* FAQ Item 1 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-[#97E2F9]">
                                <h3 className="text-xl font-bold text-gray-800 mb-2">
                                    How effective are mnemonic techniques for language learning?
                                </h3>
                                <p className="text-gray-600">
                                    Studies show that using mnemonics can improve vocabulary retention by up to 50% compared to rote memorization methods.
                                </p>
                            </div>

                            {/* FAQ Item 2 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-[#F6CBAF]">
                                <h3 className="text-xl font-bold text-gray-800 mb-2">
                                    How does the AI create personalized mnemonics?
                                </h3>
                                <p className="text-gray-600">
                                    Our AI analyzes pronunciation, meaning, and cultural context to create memorable associations that are tailored to your learning style.
                                </p>
                            </div>

                            {/* FAQ Item 3 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-[#97E2F9]">
                                <h3 className="text-xl font-bold text-gray-800 mb-2">
                                    Can I use MemoLang on my mobile device?
                                </h3>
                                <p className="text-gray-600">
                                    Yes! Our platform works on all devices with a responsive design for learning on the go.
                                </p>
                            </div>
                        </div>
                    </div>
                </section >

                {/* 6. FINAL CTA SECTION */}
                < section className="py-16 bg-[#97E2F9]" >
                    <div className="container mx-auto px-6 text-center">
                        <h2 className="text-3xl font-bold text-gray-800 mb-6">
                            Start Your Language Learning Journey Today
                        </h2>
                        <button className="px-8 py-4 rounded-full bg-[#FF8A5B] text-white font-medium text-lg hover:bg-[#ff7a43] transition-colors inline-block">
                            Sign Up Free
                        </button>
                    </div>
                </section >

                {/* 7. FOOTER */}
                < footer className="bg-gray-800 text-white py-12" >
                    <div className="container mx-auto px-6">
                        <div className="flex flex-col md:flex-row justify-between mb-8">
                            <div className="mb-6 md:mb-0">
                                <div className="flex items-center mb-4">
                                    <div className="w-8 h-8 rounded-full bg-[#97E2F9] mr-2"></div>
                                    <span className="text-xl font-bold">MemoLang</span>
                                </div>
                                <p className="text-gray-400 max-w-xs">
                                    Learn languages faster with AI-powered mnemonic flashcards.
                                </p>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-3 gap-8">
                                <div>
                                    <h3 className="text-lg font-semibold mb-4">Product</h3>
                                    <ul className="space-y-2">
                                        <li><a href="#" className="text-gray-400 hover:text-white">Features</a></li>
                                        <li><a href="#" className="text-gray-400 hover:text-white">Pricing</a></li>
                                        <li><a href="#" className="text-gray-400 hover:text-white">Languages</a></li>
                                    </ul>
                                </div>

                                <div>
                                    <h3 className="text-lg font-semibold mb-4">Company</h3>
                                    <ul className="space-y-2">
                                        <li><a href="#" className="text-gray-400 hover:text-white">About</a></li>
                                        <li><a href="#" className="text-gray-400 hover:text-white">Blog</a></li>
                                        <li><a href="#" className="text-gray-400 hover:text-white">Contact</a></li>
                                    </ul>
                                </div>

                                <div>
                                    <h3 className="text-lg font-semibold mb-4">Legal</h3>
                                    <ul className="space-y-2">
                                        <li><a href="#" className="text-gray-400 hover:text-white">Terms</a></li>
                                        <li><a href="#" className="text-gray-400 hover:text-white">Privacy</a></li>
                                        <li><a href="#" className="text-gray-400 hover:text-white">Cookies</a></li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div className="border-t border-gray-700 pt-8">
                            <div className="flex flex-col md:flex-row justify-between items-center">
                                <p className="text-gray-400 text-sm mb-4 md:mb-0">
                                    © 2025 MemoLang. All rights reserved.
                                </p>
                                <div className="flex space-x-4">
                                    <a href="#" className="text-gray-400 hover:text-white">
                                        <span className="sr-only">Facebook</span>
                                        <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">f</div>
                                    </a>
                                    <a href="#" className="text-gray-400 hover:text-white">
                                        <span className="sr-only">Twitter</span>
                                        <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">t</div>
                                    </a>
                                    <a href="#" className="text-gray-400 hover:text-white">
                                        <span className="sr-only">Instagram</span>
                                        <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">i</div>
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                </footer >
            </div >
        </>
    );
};

export default LandingPage;