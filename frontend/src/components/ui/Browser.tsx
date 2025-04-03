import React, { ReactNode, CSSProperties } from 'react';

interface BrowserProps {
    children: ReactNode;
    className?: string;
    showUrlBar?: boolean;
    urlText?: string;
    dark?: boolean;
}

const Browser: React.FC<BrowserProps> = ({
    children,
    className = "",
    showUrlBar = true,
    urlText = "example.com",
    dark = false
}) => {
    // Conditional classes based on dark mode
    const containerClasses = dark
        ? "border-gray-700 shadow-lg"
        : "border-gray-300 shadow-lg";

    const headerClasses = dark
        ? "bg-gray-800"
        : "bg-gray-200";

    const urlBarClasses = dark
        ? "bg-gray-700"
        : "bg-gray-100";

    const urlTextClasses = dark
        ? "text-gray-300"
        : "text-gray-500";

    const contentClasses = dark
        ? "bg-gray-900"
        : "bg-white";

    return (
        <div className={`rounded-lg overflow-hidden border ${containerClasses} ${className}`}
        >
            {/* Browser header */}
            <div className={`${headerClasses} h-8 flex items-center px-4 relative`}>
                {/* Traffic light dots - these stay the same in dark mode */}
                <div className="flex space-x-2">
                    <span className="h-3 w-3 bg-red-500 rounded-full flex items-center justify-center">
                        <span className="h-2 w-2 bg-red-600 rounded-full opacity-0 group-hover:opacity-100"></span>
                    </span>
                    <span className="h-3 w-3 bg-yellow-500 rounded-full"></span>
                    <span className="h-3 w-3 bg-green-500 rounded-full"></span>
                </div>

                {/* Optional: URL bar */}
                {showUrlBar && (
                    <div className={`absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 
                         ${urlBarClasses} rounded-md h-5 w-1/2 flex items-center justify-center`}>
                        <span className={`text-xs ${urlTextClasses} truncate`}>{urlText}</span>
                    </div>
                )}
            </div>

            {/* Browser content */}
            <div className={contentClasses}>
                {children}
            </div>
        </div>
    );
};

export default Browser;