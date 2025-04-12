"use client"

import React, { useRef } from "react";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import Button from "@/components/ui/Button";
import { motion } from "framer-motion";
import { useInView } from "framer-motion";

const FooterSection = () => {
    const ref = useRef(null);
    const isInView = useInView(ref, { once: false, amount: 0.3 });




    return (
        <>
            {/* Shadow line with animation */}
            <div ref={ref} className="relative w-full h-[4rem] flex justify-center overflow-hidden">
                <motion.span
                    className="absolute shadow-lg shadow-blue-500/50 border-t border-gradient-to-r from-blue-500 to-teal-400"
                    style={{
                        boxShadow: " 0 0 40px 10px rgba(56, 191, 248, 0.9),  0 0 40px 10px rgba(45, 212, 191, 0.9)",
                    }}
                    initial={{ width: "0%" }}
                    animate={isInView ? { width: "100%" } : { width: "0%" }}
                    transition={{ duration: 1.2, ease: "easeOut" }}
                />
            </div>

            <div className="container mx-auto px-6 text-center mb-10" >
                <h2 className="text-3xl font-bold text-gray-800 mb-6">
                    <TextGenerateEffect words={'Start Your Language Learning Journey Today'} className="text-3xl bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent" />
                </h2>
                <motion.div
                    initial={{ y: "50%", opacity: "0%" }}
                    animate={isInView ? { y: "0%", opacity: "100%" } : { y: "50%", opacity: "0%" }}
                    transition={{ duration: 1, ease: "easeOut" }}

                >
                    <Button text="Sign Up Free" />
                </motion.div>

            </div>
            <footer className="bg-gray-800 text-white py-12">
                <div className="container mx-auto px-6">
                    <div className="flex flex-col md:flex-row justify-between mb-8">
                        <div className="mb-6 md:mb-0">
                            <div className="flex items-center mb-4">
                                <span className="text-xl font-bold">mnemorai</span>
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
                                © 2025 mnemorai. All rights reserved.
                            </p>
                            <div className="flex space-x-4">

                            </div>
                        </div>
                    </div>
                </div>
            </footer>
        </>
    )
}

export default FooterSection;