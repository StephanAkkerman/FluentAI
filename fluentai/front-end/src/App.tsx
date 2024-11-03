import React, { useState, useEffect } from "react";

import "./App.css";

// pages
import Header from "./pages/Header";
import LoadingPage from "./pages/LoadingPage";

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
    <div className="App">
      <Header onGearClick={handleGearClick} />
      {initLoad && <LoadingPage isLoading={initLoad} />}
    </div>
  );
}

export default App;
