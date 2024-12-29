import React, { useState, useEffect } from "react";

interface APIStatus {
  available: boolean | null;
  name: string;
  description: string;
  show: boolean;
}

export default function StatusChecker() {
  const [statuses, setStatuses] = useState<APIStatus[]>([
    { name: "AnkiConnect", description: "Anki synchronization", available: null, show: true },
    { name: "Card Generator", description: "Card creation API", available: null, show: true },
  ]);

  useEffect(() => {
    const checkAPIStatuses = async () => {
      const newStatuses = await Promise.all(
        statuses.map(async (status) => {
          try {
            if (status.name === "AnkiConnect") {
              const response = await fetch("http://127.0.0.1:8765", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "version", version: 6 }),
              });
              if (response.ok) {
                const data = await response.json();
                return { ...status, available: !!data.result, show: true };
              }
            }
            if (status.name === "Card Generator") {
              const response = await fetch("http://localhost:8000/create_card/supported_languages");
              if (response.ok) {
                return { ...status, available: true, show: true };
              }
            }
            return { ...status, available: false, show: true };
          } catch (error) {
            console.error(`Error checking ${status.name}:`, error);
            return { ...status, available: false, show: true };
          }
        })
      );
      setStatuses(newStatuses);

      newStatuses.forEach((status, index) => {
        if (status.available === true) {
          setTimeout(() => {
            setStatuses((prev) => {
              const updated = [...prev];
              updated[index] = { ...status, show: false };
              return updated;
            });
          }, 3000);
        }
      });
    };
    checkAPIStatuses();
  }, []);

  return (
    <div className="space-y-0">
      {statuses.map((status, index) => (
        <div
          key={index}
          className={`transform transition-all duration-500 ease-in-out overflow-hidden
            ${status.show ? "max-h-24 mb-4 opacity-100 translate-y-0" : "max-h-0 mb-0 opacity-0 -translate-y-4"}
            ${status.available === null
              ? "bg-yellow-100 text-yellow-800"
              : status.available
                ? "bg-green-100 text-green-800"
                : "bg-red-100 text-red-800"
            }`}
        >
          <div className="p-4 text-center rounded">
            {status.available === null && <p>Checking {status.name} availability...</p>}
            {status.available === true && (
              <p>{status.name} is available! ({status.description})</p>
            )}
            {status.available === false && (
              <p>
                <strong>{status.name} is unavailable.</strong> Ensure the {status.description} is running and accessible.
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
