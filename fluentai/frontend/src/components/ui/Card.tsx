import React from "react";

interface CardProps {
  title: string;
  children?: React.ReactNode;
}

export default function Card({ title, children }: CardProps) {
  return (
    <div className="bg-white p-4 rounded-md shadow-md">
      <h2 className="text-xl font-semibold">{title}</h2>
      {children}
    </div>
  );
}

