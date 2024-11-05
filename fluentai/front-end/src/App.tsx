import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import "./App.css";

// pages
import Header from "./pages/Header";
import LoadingPage from "./pages/LoadingPage";
import CardCreation from "./pages/CardCreation";

function App() {
  const [openSettings, setOpenSettings] = useState(true);
  const [initLoad, setInitLoad] = useState(true);

  useEffect(() => {
    // Simulate loading (e.g., fetching data)
    const timer = setTimeout(() => {
      setInitLoad(false);
    }, 3000); // Adjust the time as needed

    return () => clearTimeout(timer);
  }, []);

  // Create a function to handle the gear click
  const handleGearClick = () => {
    setOpenSettings((prev) => !prev);
  };

  return (
    <Router basename="/FluentAI">
      {/* basename should be changed to / when we have own domain */}
      <div className="App">
        <Header onGearClick={handleGearClick} />
        {initLoad && <LoadingPage isLoading={initLoad} />}
        <Routes>
          <Route path="/" element={<CardCreation />} />
          {/* Add more routes as needed */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;
