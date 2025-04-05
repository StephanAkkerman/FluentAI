"use client"
import React from "react";
import { FloatingDock } from "@/components/ui/floating-dock";
import Image from "next/image";

export function LanguageDock() {
    const languages = [
        {
            title: "Hello",
            language: "English",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image
                            src="/flags/gb.svg"
                            alt="UK Flag"
                            fill
                            sizes="100%"
                            className="object-cover"
                        />
                    </div>
                </div>
            ),
            href: "#",
        },
        {
            title: "Hola",
            language: "Spanish",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image
                            src="/flags/es.svg"
                            alt="Spain Flag"
                            fill
                            sizes="100%"
                            className="object-cover"
                        />
                    </div>
                </div>
            ),
            href: "#",
        },
        {
            title: "Bonjour",
            language: "French",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image
                            src="/flags/fr.svg"
                            alt="France Flag"
                            fill
                            sizes="100%"
                            className="object-cover"
                        />
                    </div>
                </div>
            ),
            href: "#",
        },
        {
            title: "Ciao",
            language: "Italian",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image
                            src="/flags/it.svg"
                            alt="Italy Flag"
                            fill
                            sizes="100%"
                            className="object-cover"
                        />
                    </div>
                </div>
            ),
            href: "#",
        },
        {
            title: "Hallo",
            language: "German",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image
                            src="/flags/de.svg"
                            alt="Germany Flag"
                            fill
                            sizes="100%"
                            className="object-cover"
                        />
                    </div>
                </div>
            ),
            href: "#",
        },
        {
            title: "こんにちは",
            language: "Japanese",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image
                            src="/flags/jp.svg"
                            alt="Japan Flag"
                            fill
                            sizes="100%"
                            className="object-cover"
                        />
                    </div>
                </div>
            ),
            href: "#",
        },
        {
            title: "你好",
            language: "Chinese",
            icon: (
                <div className="relative flex items-center justify-center h-full w-full">
                    <div className="absolute inset-0 rounded-full overflow-hidden">
                        <Image
                            src="/flags/cn.svg"
                            alt="China Flag"
                            fill
                            sizes="100%"
                            className="object-cover"
                        />
                    </div>
                </div>
            ),
            href: "#",
        },
    ];

    return (
        <FloatingDock
            items={languages}
            mobileClassName="translate-y-20" // only for demo, remove for production
        />
    );
}

export default LanguageDock;