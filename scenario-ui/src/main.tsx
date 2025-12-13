import React from "react";
import ReactDOM from "react-dom/client";
import ScenarioApp from "./ScenarioApp";
import "./index.css";

const rootElement = document.getElementById("root");

if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <ScenarioApp />
    </React.StrictMode>
  );
} else {
  console.error("Root element not found");
}
