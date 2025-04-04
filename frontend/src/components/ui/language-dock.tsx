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
                <div className="relative h-5 w-8 overflow-hidden rounded-sm">
                    <Image
                        src="/flags/gb.svg"
                        alt="UK Flag"
                        fill
                        className="object-cover"
                    />
                </div>
            ),
            href: "#",
        },
        {
            title: "Hola",
            language: "Spanish",
            icon: (
                <div className="relative h-5 w-8 overflow-hidden rounded-sm">
                    <Image
                        src="/flags/es.svg"
                        alt="Spain Flag"
                        fill
                        className="object-cover"
                    />
                </div>
            ),
            href: "#",
        },
        {
            title: "Bonjour",
            language: "French",
            icon: (
                <div className="relative h-5 w-8 overflow-hidden rounded-sm">
                    <Image
                        src="/flags/fr.svg"
                        alt="France Flag"
                        fill
                        className="object-cover"
                    />
                </div>
            ),
            href: "#",
        },
        {
            title: "Ciao",
            language: "Italian",
            icon: (
                <div className="relative h-5 w-8 overflow-hidden rounded-sm">
                    <Image
                        src="/flags/it.svg"
                        alt="Italy Flag"
                        fill
                        className="object-cover"
                    />
                </div>
            ),
            href: "#",
        },
        {
            title: "Hallo",
            language: "German",
            icon: (
                <div className="relative h-5 w-8 overflow-hidden rounded-sm">
                    <Image
                        src="/flags/de.svg"
                        alt="Germany Flag"
                        fill
                        className="object-cover"
                    />
                </div>
            ),
            href: "#",
        },
        {
            title: "こんにちは",
            language: "Japanese",
            icon: (
                <div className="relative h-5 w-8 overflow-hidden rounded-sm">
                    <Image
                        src="/flags/jp.svg"
                        alt="Japan Flag"
                        fill
                        className="object-cover"
                    />
                </div>
            ),
            href: "#",
        },
        {
            title: "你好",
            language: "Chinese",
            icon: (
                <div className="relative h-5 w-8 overflow-hidden rounded-sm">
                    <Image
                        src="/flags/cn.svg"
                        alt="China Flag"
                        fill
                        className="object-cover"
                    />
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