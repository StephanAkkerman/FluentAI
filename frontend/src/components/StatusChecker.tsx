"use client";

import { useEffect, useRef } from "react";
import { ANKI_CONFIG } from "@/config/constants";
import { useToast } from "@/contexts/ToastContext";

interface APIStatus {
  available: boolean | null;
  name: string;
  description: string;
}

export default function StatusChecker() {
  const { showToast, hideAllToasts } = useToast();
  const statusesRef = useRef<APIStatus[]>([
    { name: "AnkiConnect", description: "Anki synchronization", available: null },
    { name: "Card Generator", description: "Card creation API", available: null },
  ]);

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
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
  const POLLING_INTERVAL = 5000; // 5 seconds between checks
  const SUCCESS_DISPLAY_TIME = 3000; // 3 seconds to display success

  useEffect(() => {
    // Function to check all statuses at once
    const checkAllStatuses = async () => {
      if (!isMounted.current) return;

      // Create a copy to work with
      const currentStatuses = [...statusesRef.current];
      const updatedStatuses = [...currentStatuses];
      const recoveredServices: string[] = [];
      let hasServiceDown = false;
      let statusChanged = false;

      // Check each service
      for (let i = 0; i < updatedStatuses.length; i++) {
        const status = updatedStatuses[i];
        const isAvailable = await checkAPIStatus(status.name);

        // If status changed from not available to available (true transition)
        if (status.available === false && isAvailable) {
          updatedStatuses[i] = { ...status, available: true };
          recoveredServices.push(status.name);
          statusChanged = true;
        }
        // If checking for the first time and it's available
        else if (status.available === null && isAvailable) {
          updatedStatuses[i] = { ...status, available: true };
        }
        // If service is unavailable
        else if (!isAvailable) {
          if (status.available !== false) {
            statusChanged = true;
          }
          updatedStatuses[i] = { ...status, available: false };
          hasServiceDown = true;
        }
      }

      // Update statuses ref
      statusesRef.current = updatedStatuses;

      // Show appropriate toast notifications
      if (statusChanged) {
        // Clear existing toasts when status changes
        hideAllToasts();

        // If any service was recovered, show a success toast for it
        if (recoveredServices.length > 0) {
          // Show success toast for specifically recovered services
          showToast({
            type: 'success',
            title: `Service${recoveredServices.length > 1 ? 's' : ''} Recovered`,
            message: `${recoveredServices.join(", ")} ${recoveredServices.length > 1 ? 'are' : 'is'} now available.`,
            duration: SUCCESS_DISPLAY_TIME // Show longer so users notice it
          });
        }

        // If we still have services down, show an error toast after a brief delay
        // This prevents the toasts from appearing simultaneously
        if (hasServiceDown) {
          setTimeout(() => {
            if (!isMounted.current) return;

            const unavailableServices = statusesRef.current
              .filter(s => s.available === false)
              .map(s => s.name)
              .join(", ");

            showToast({
              type: 'error',
              title: 'Service Interruption',
              message: `${unavailableServices} ${statusesRef.current.filter(s => s.available === false).length > 1 ? 'are' : 'is'} unavailable. Please check your connections.`,
              duration: 0 // Keep until resolved
            });
          }, recoveredServices.length > 0 ? 300 : 0); // Small delay if we just showed a recovery toast
        } else if (recoveredServices.length > 0) {
          // If all services are up after some were down, show an additional "all clear" notification
          setTimeout(() => {
            if (!isMounted.current) return;

            showToast({
              type: 'info',
              title: 'All Systems Operational',
              message: 'All services are now running properly.',
              duration: SUCCESS_DISPLAY_TIME
            });
          }, 300); // Small delay after the recovery toast
        }
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
  }, [showToast, hideAllToasts]); // Include toast functions as dependencies

  // This component doesn't render anything visible on its own now
  return null;
}
