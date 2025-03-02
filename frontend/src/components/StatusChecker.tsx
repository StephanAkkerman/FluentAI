import React, { useState, useEffect, useRef } from "react";
import { ANKI_CONFIG } from "@/config/constants";

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

  // Use a single ref to store the polling interval
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Keep track of whether component is mounted
  const isMounted = useRef(true);

  // Define the API status check function
  const checkAPIStatus = async (name: string): Promise<boolean> => {
    try {
      if (name === "AnkiConnect") {
        const response = await fetch(ANKI_CONFIG.API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "version", version: 6 }),
        });
        if (response.ok) {
          const data = await response.json();
          return !!data.result;
        }
      }
      if (name === "Card Generator") {
        const response = await fetch("http://localhost:8000/create_card/supported_languages");
        if (response.ok) {
          return true;
        }
      }
      return false;
    } catch (error) {
      console.error(`Error checking ${name}:`, error);
      return false;
    }
  };

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      isMounted.current = false;
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, []);

  // Set up the polling mechanism
  const POLLING_INTERVAL = 1000; // 10 seconds between checks
  const SUCCESS_DISPLAY_TIME = 3000; // 3 seconds to display success

  useEffect(() => {
    // Function to check all statuses at once
    const checkAllStatuses = async () => {
      if (!isMounted.current) return;

      // Create a copy to work with
      const updatedStatuses = [...statuses];
      let shouldPoll = false;

      // Check each service
      for (let i = 0; i < updatedStatuses.length; i++) {
        const status = updatedStatuses[i];
        const isAvailable = await checkAPIStatus(status.name);

        // If status changed from not available to available (true transition)
        if (status.available === false && isAvailable) {
          updatedStatuses[i] = { ...status, available: true, show: true };

          // Set a timeout to hide the status after SUCCESS_DISPLAY_TIME
          if (isMounted.current) {
            setTimeout(() => {
              if (isMounted.current) {
                setStatuses(current => {
                  const next = [...current];
                  next[i] = { ...next[i], show: false };
                  return next;
                });
              }
            }, SUCCESS_DISPLAY_TIME);
          }
        }
        // If checking for the first time and it's available
        else if (status.available === null && isAvailable) {
          updatedStatuses[i] = { ...status, available: true, show: true };

          // Set a timeout to hide the status after SUCCESS_DISPLAY_TIME
          if (isMounted.current) {
            setTimeout(() => {
              if (isMounted.current) {
                setStatuses(current => {
                  const next = [...current];
                  next[i] = { ...next[i], show: false };
                  return next;
                });
              }
            }, SUCCESS_DISPLAY_TIME);
          }
        }
        // If status is already available, maintain its state without showing again
        else if (status.available === true && isAvailable) {
          // Keep existing state, don't change 'show' property
          updatedStatuses[i] = { ...status, available: true };
        }
        // If status is unavailable
        else if (!isAvailable) {
          updatedStatuses[i] = { ...status, available: false, show: true };
          shouldPoll = true; // We need to keep polling
        }
      }

      // Update state with all changes at once
      if (isMounted.current) {
        setStatuses(updatedStatuses);
      }

      // If all services are now available, we can stop polling
      if (!shouldPoll && pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };

    // Run initial check
    checkAllStatuses();

    // Set up single polling interval (only if not already set)
    if (!pollingIntervalRef.current) {
      pollingIntervalRef.current = setInterval(checkAllStatuses, POLLING_INTERVAL);
    }

    // Cleanup on effect change
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, []); // Empty dependency array - only run on mount

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
