"use client"
import WhatSection from "./landing-page/WhatSection"
import WhySection from "./landing-page/WhySection";
import HowSection from "./landing-page/HowSection";
import HeroSection from "./landing-page/HeroSection";
import AIPage from "./landing-page/AISection";
import FAQSection from "./landing-page/FAQSection";
import React from "react";



const LandingPage = () => {


    return (
        <>


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
                < section id="features" className="py-20" >
                    <AIPage />
                </section >



                {/* 5. FAQ SECTION */}
                < section id="faq" >
                    <FAQSection />
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