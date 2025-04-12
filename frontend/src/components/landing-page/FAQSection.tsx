"use client"

import React, { useState } from "react"

const FAQSection = () => {
    // State to track which FAQ items are expanded
    const [expandedItems, setExpandedItems] = useState<{ [key: number]: boolean }>({});

    // FAQ data - doubled to 10 items
    const faqItems = [
        {
            question: "How effective are mnemonic techniques for language learning?",
            answer: "Studies show that using mnemonics can improve vocabulary retention by up to 50% compared to rote memorization methods. This is because mnemonics create stronger neural connections by linking new information to existing knowledge."
        },
        {
            question: "How does the AI create personalized mnemonics?",
            answer: "Our AI analyzes pronunciation, meaning, and cultural context to create memorable associations that are tailored to your learning style. It also takes into account your native language and previous learning patterns."
        },
        {
            question: "Can I use MemoLang on my mobile device?",
            answer: "Yes! Our platform works on all devices with a responsive design for learning on the go. We have native apps for iOS and Android, as well as a mobile-optimized web version."
        },
        {
            question: "How many languages does MemoLang support?",
            answer: "We currently support 12 major languages including Spanish, French, German, Mandarin, Japanese, Korean, Italian, Portuguese, Russian, Arabic, Hindi, and English."
        },
        {
            question: "Is there a free trial available?",
            answer: "Yes! We offer a 7-day free trial with access to all features so you can experience the full benefits of our AI-powered learning system."
        },
        {
            question: "What makes MemoLang different from other language apps?",
            answer: "Unlike traditional apps that focus on repetition, MemoLang uses AI to create personalized memory techniques based on cognitive science. Our approach adapts to your learning style and creates connections between words that stick in your long-term memory."
        },
        {
            question: "How long does it take to see results?",
            answer: "Most users report significant improvements in vocabulary retention within the first two weeks. Our internal studies show that after 30 days of consistent use, users typically remember 3x more words compared to traditional methods."
        },
        {
            question: "Can I import my own vocabulary lists?",
            answer: "Absolutely! You can upload custom vocabulary lists in CSV format or copy-paste them directly. Our AI will automatically generate mnemonics for your custom words."
        },
        {
            question: "Do you offer group or classroom plans?",
            answer: "Yes, we offer special plans for educators and language schools. Our classroom dashboard allows teachers to track student progress and assign custom learning paths. Contact our sales team for educational pricing."
        },
        {
            question: "How do I cancel my subscription?",
            answer: "You can cancel your subscription anytime from your account settings. If you cancel during your free trial, you won't be charged. If you cancel after being billed, you'll maintain access until the end of your billing period."
        }
    ];

    // Toggle expanded state for a specific FAQ item
    const toggleExpanded = (index: number) => {
        setExpandedItems(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    };

    return (
        <div className="container mx-auto px-6 py-12">
            <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">
                Frequently Asked Questions
            </h2>

            {/* Scrollable FAQ container with custom scrollbar */}
            <div className="max-w-3xl mx-auto max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                <div className="border-t border-gray-200">
                    {faqItems.map((item, index) => (
                        <div
                            key={index}
                            className="border-b border-gray-200"
                        >
                            <button
                                onClick={() => toggleExpanded(index)}
                                className="w-full flex justify-between items-center py-5 px-2 text-left focus:outline-none focus:ring-0"
                            >
                                <h3
                                    className={`text-lg font-medium transition-colors duration-300 ${expandedItems[index]
                                            ? 'bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent'
                                            : 'text-gray-800'
                                        }`}
                                >
                                    {item.question}
                                </h3>
                                <svg
                                    className={`w-5 h-5 transition-transform duration-300 ${expandedItems[index]
                                            ? 'rotate-180 text-teal-400'
                                            : 'text-gray-500'
                                        }`}
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M19 9l-7 7-7-7"
                                    />
                                </svg>
                            </button>

                            {/* Answer section with slide animation */}
                            <div
                                className={`overflow-hidden transition-all duration-300 ease-in-out ${expandedItems[index]
                                        ? 'max-h-40 opacity-100 pb-5 px-2'
                                        : 'max-h-0 opacity-0'
                                    }`}
                            >
                                <p className="text-gray-600">
                                    {item.answer}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Custom scrollbar styling */}
            <style jsx global>{`
                /* Customize scrollbar for webkit browsers */
                .custom-scrollbar::-webkit-scrollbar {
                    width: 8px;
                }
                
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background-color: #9CA3AF; /* Gray-500 */
                    border-radius: 20px;
                }
                
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(to bottom, #3B82F6, #2DD4BF); /* blue-500 to teal-400 */
                }
                
                /* For Firefox */
                .custom-scrollbar {
                    scrollbar-width: thin;
                    scrollbar-color: #9CA3AF transparent;
                }
                
                /* Make sure expanded answers can be taller for longer content */
                .max-h-40 {
                    max-height: 160px; /* Increased from default */
                }
            `}</style>
        </div>
    )
}

export default FAQSection