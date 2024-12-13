import { useState } from "react";

interface AutoCompleteInputProps {
  suggestions: string[];
  onSelect: (selected: string) => void;
}

export default function AutoCompleteInput({
  suggestions,
  onSelect,
}: AutoCompleteInputProps) {
  const [inputValue, setInputValue] = useState("");
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);

  const handleChange = (value: string) => {
    setInputValue(value);
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
    <div className="relative">
      <input
        type="text"
        value={inputValue}
        onChange={(e) => handleChange(e.target.value)}
        className="border rounded p-2 w-full"
        placeholder="Choose a language"
      />
      {filteredSuggestions.length > 0 && (
        <ul className="absolute left-0 w-full border bg-white mt-1 z-10 max-h-60 overflow-y-auto rounded">
          {filteredSuggestions.map((s, idx) => (
            <li
              key={idx}
              className="p-2 hover:bg-gray-200 cursor-pointer"
              onClick={() => handleSelect(s)}
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

