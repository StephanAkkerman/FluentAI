import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import "./App.css";

// pages
import Header from "./pages/Header";
import SideNav from "./pages/SideNav";
import LoadingPage from "./pages/LoadingPage";
import CardCreation from "./pages/CardCreation";

function App() {
  const [sideNav, setSideNav] = useState(true);
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
    setSideNav((prev) => !prev);
  };

  return (
    <Router basename="/FluentAI">
      {/* basename should be changed to / when we have own domain */}
      <div className="App">

        <SideNav show={sideNav} onCloseClick={handleGearClick} />

        <Header onGearClick={handleGearClick} />
        {initLoad && <LoadingPage isLoading={initLoad} />}
        <Routes>
          <Route path="/" element={<CardCreation />} />
          <Route path="/CardCreation" element={<CardCreation />} />
          <Route path="/CardLibrary" element={<></>} />
          <Route path="/Settings" element={<></>} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
