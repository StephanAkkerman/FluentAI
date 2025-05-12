import { useState, useEffect } from "react";

interface AutoCompleteInputProps {
  suggestions: string[];
  onSelect: (selected: string) => void;
  initialValue?: string;
  placeholder?: string;
}

export default function AutoCompleteInput({
  suggestions,
  onSelect,
  initialValue = "",
  placeholder = "Type to search..."
}: AutoCompleteInputProps) {
  const [inputValue, setInputValue] = useState(initialValue);
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => {
    setInputValue(initialValue);
  }, [initialValue]);

  const handleChange = (value: string) => {
    setInputValue(value);
    onSelect(value); // Allow freeform input
    setFilteredSuggestions(
      suggestions.filter((s) => s.toLowerCase().includes(value.toLowerCase()))
    );
  };

  const handleSelect = (suggestion: string) => {
    setInputValue(suggestion);
    setFilteredSuggestions([]);
    onSelect(suggestion);
  };

  return (
    <div className="relative" onBlur={() => setIsFocused(false)} onFocus={() => setIsFocused(true)}>
      <input
        type="text"
        value={inputValue}
        onChange={(e) => handleChange(e.target.value)}
        //className="border rounded p-2 w-full bg-white text-gray-800 dark:text-black-200 dark:border-gray-600"
        className="border rounded-lg p-2 w-full bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 
                         focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
        placeholder={placeholder}
      />
      {isFocused && filteredSuggestions.length > 0 && (
        <ul className="absolute left-0 w-full border mt-1 z-10 max-h-60 overflow-y-auto rounded shadow-lg bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600">
          {filteredSuggestions.map((s, idx) => (
            <li
              key={idx}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-900 cursor-pointer text-gray-800 dark:text-white"
              onMouseDown={() => handleSelect(s)}
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
