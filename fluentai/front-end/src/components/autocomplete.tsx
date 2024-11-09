import React, { FC, useState, useRef, useEffect } from "react";
import "../styles/components/AutoComplete.css";
import FormField from "./formfield";

interface AutocompleteInputProps {
  suggestions: string[];
  onSelect: (name: string) => void;
  formSubmitted?: boolean;
  title?: string;
}

const AutocompleteInput: FC<AutocompleteInputProps> = ({
  suggestions,
  onSelect,
  title,
  formSubmitted,
}) => {
  const [inputValue, setInputValue] = useState<string>("");
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);

  const autocompleteRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        autocompleteRef.current &&
        !autocompleteRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false); // Close dropdown if clicked outside
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleChange = (userInput: string) => {
    const newFilteredSuggestions = suggestions.filter(
      (suggestion) =>
        suggestion.toLowerCase().indexOf(userInput.toLowerCase()) === 0
    );
    setInputValue(userInput);
    setFilteredSuggestions(newFilteredSuggestions);
    setShowSuggestions(true);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
    setShowSuggestions(false);
    onSelect(suggestion);
  };

  const handleDropdownClick = () => {
    setShowSuggestions(!showSuggestions);
    setFilteredSuggestions(suggestions);
  };

  let suggestionsListComponent;

  if (showSuggestions && filteredSuggestions.length) {
    suggestionsListComponent = (
      <ul className="suggestions">
        {filteredSuggestions.map((suggestion, index) => (
          <li key={index} onClick={() => handleSuggestionClick(suggestion)}>
            {suggestion}
          </li>
        ))}
      </ul>
    );
  }

  return (
    <div className="autocomplete-input" ref={autocompleteRef}>
      {title !== "" && <div className="title">{title}</div>}
      <div className="selector">
        <FormField
          required={true}
          value={inputValue}
          id="name"
          label=""
          onChange={handleChange}
          formSubmitted={formSubmitted}
        />
        <button onClick={handleDropdownClick}>▼</button>
        {suggestionsListComponent}
      </div>
    </div>
  );
};

export default AutocompleteInput;
