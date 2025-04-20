// components/LanguageDock.tsx
"use client";
import React, { useState, useLayoutEffect } from "react";
import { FloatingDock } from "@/components/ui/floating-dock";
import Image from "next/image";

interface Language {
    title: string;
    language: string;
    icon: React.ReactNode;
}

interface LanguageDockProps {
    className?: string;
}

export function LanguageDock({ className }: LanguageDockProps) {
    // 1) track window size
    const [windowSize, setWindowSize] = useState<{ width: number; height: number }>(() => ({
        width: typeof window !== "undefined" ? window.innerWidth : 1024,
        height: typeof window !== "undefined" ? window.innerHeight : 768,
    }));

    useLayoutEffect(() => {
        function handleResize() {
            setWindowSize({ width: window.innerWidth, height: window.innerHeight });
        }
        window.addEventListener("resize", handleResize);
        handleResize();
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    // 2) breakpoints
    const isSmallScreen = windowSize.width < 640;

    // 3) your languages
    const languages: Language[] = [
        {
            title: "Hello",
            language: "English",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image src="/flags/gb.svg" alt="UK Flag" fill sizes="100%" className="object-cover" />
                    </div>
                </div>
            ),
        },
        {
            title: "Hola",
            language: "Spanish",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image src="/flags/es.svg" alt="Spain Flag" fill sizes="100%" className="object-cover" />
                    </div>
                </div>
            ),
        },
        {
            title: "Bonjour",
            language: "French",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image src="/flags/fr.svg" alt="France Flag" fill sizes="100%" className="object-cover" />
                    </div>
                </div>
            ),
        },
        {
            title: "Ciao",
            language: "Italian",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image src="/flags/it.svg" alt="Italy Flag" fill sizes="100%" className="object-cover" />
                    </div>
                </div>
            ),
        },
        {
            title: "Hallo",
            language: "German",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image src="/flags/de.svg" alt="Germany Flag" fill sizes="100%" className="object-cover" />
                    </div>
                </div>
            ),
        },
        {
            title: "こんにちは",
            language: "Japanese",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image src="/flags/jp.svg" alt="Japan Flag" fill sizes="100%" className="object-cover" />
                    </div>
                </div>
            ),
        },
        {
            title: "你好",
            language: "Chinese",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image src="/flags/cn.svg" alt="China Flag" fill sizes="100%" className="object-cover" />
                    </div>
                </div>
            ),
        },
    ];

    // 4) split into two roughly equal arrays
    const mid = Math.ceil(languages.length / 2);
    const firstHalf = languages.slice(0, mid);
    const secondHalf = languages.slice(mid);

    return (
        <div className={className}>
            {isSmallScreen ? (
                <div className="flex flex-col items-center justify-center">
                    <FloatingDock items={firstHalf} mobileClassName="translate-y-20" />
                    <FloatingDock items={secondHalf} mobileClassName="translate-y-20" />
                </div>
            ) : (
                <FloatingDock items={languages} mobileClassName="translate-y-20" />
            )}
        </div>
    );
}

export default LanguageDock;
