import React, { useState, useEffect } from 'react';
import { X, AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

interface ToastProps {
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
  onClose: () => void;
  isVisible: boolean;
}

export const Toast: React.FC<ToastProps> = ({
  type,
  title,
  message,
  duration = 5000,
  onClose,
  isVisible
}) => {
  const [isClosing, setIsClosing] = useState(false);

  // Auto-close functionality
  useEffect(() => {
    if (!isVisible || duration === 0) return;

    const timer = setTimeout(() => {
      setIsClosing(true);
      setTimeout(onClose, 300); // Wait for animation to complete
    }, duration);

    return () => clearTimeout(timer);
  }, [isVisible, duration, onClose]);

  // Handle manual close
  const handleClose = () => {
    setIsClosing(true);
    setTimeout(onClose, 300); // Wait for animation to complete
  };

  if (!isVisible) return null;

  // Define icon and styles based on toast type
  const getIconAndStyles = () => {
    switch (type) {
      case 'success':
        return {
          icon: <CheckCircle className="h-5 w-5" />,
          bgClass: 'bg-green-50 dark:bg-green-900/90',
          borderClass: 'border-green-200 dark:border-green-800',
          textClass: 'text-green-700 dark:text-green-200'
        };
      case 'error':
        return {
          icon: <AlertCircle className="h-5 w-5" />,
          bgClass: 'bg-red-50 dark:bg-red-900/90',
          borderClass: 'border-red-200 dark:border-red-800',
          textClass: 'text-red-700 dark:text-red-200'
        };
      case 'warning':
        return {
          icon: <AlertTriangle className="h-5 w-5" />,
          bgClass: 'bg-yellow-50 dark:bg-yellow-900/90',
          borderClass: 'border-yellow-200 dark:border-yellow-800',
          textClass: 'text-yellow-700 dark:text-yellow-200'
        };
      case 'info':
      default:
        return {
          icon: <Info className="h-5 w-5" />,
          bgClass: 'bg-blue-50 dark:bg-blue-900/90',
          borderClass: 'border-blue-200 dark:border-blue-800',
          textClass: 'text-blue-700 dark:text-blue-200'
        };
    }
  };

  const { icon, bgClass, borderClass, textClass } = getIconAndStyles();

  return (
    <div
      className={`rounded-lg shadow-lg border ${bgClass} ${borderClass} ${textClass} 
                  transition-all duration-300 transform 
                  ${isClosing ? 'opacity-0 translate-y-2' : 'opacity-100 translate-y-0'}`}
    >
      <div className="p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0 mr-3">
            {icon}
          </div>
          <div className="flex-1">
            <div className="font-medium">{title}</div>
            {message && <div className="mt-1 text-sm">{message}</div>}
          </div>
          <button
            className="ml-4 flex-shrink-0 inline-flex text-gray-400 hover:text-gray-500 dark:text-gray-300 dark:hover:text-gray-100 focus:outline-none"
            onClick={handleClose}
          >
            <span className="sr-only">Close</span>
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

// Toast Container Component for multiple toasts
export const ToastContainer: React.FC<{
  children: React.ReactNode;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}> = ({
  children,
  position = 'bottom-right'
}) => {
    const positionClasses = {
      'top-right': 'top-4 right-4',
      'top-left': 'top-4 left-4',
      'bottom-right': 'bottom-4 right-4',
      'bottom-left': 'bottom-4 left-4'
    };

    return (
      <div className={`fixed ${positionClasses[position]} z-50 w-full max-w-sm space-y-4 pointer-events-none`}>
        <div className="pointer-events-auto space-y-2">
          {children}
        </div>
      </div>
    );
  };
