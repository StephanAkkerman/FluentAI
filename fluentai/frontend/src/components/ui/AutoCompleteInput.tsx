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
        className="border rounded p-2 w-full bg-white text-gray-800 dark:text-black-200 dark:border-gray-600"
        placeholder={placeholder}
      />
      {isFocused && filteredSuggestions.length > 0 && (
        <ul className="absolute left-0 w-full border bg-white mt-1 z-10 max-h-60 overflow-y-auto rounded shadow-lg dark:border-gray-700">
          {filteredSuggestions.map((s, idx) => (
            <li
              key={idx}
              className="p-2 hover:bg-gray-200 cursor-pointer text-gray-800"
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
