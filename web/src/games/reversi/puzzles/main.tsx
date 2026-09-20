import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "../../../index.css";
import { PuzzlesScreen } from "./PuzzlesScreen";

const root = document.getElementById("root");
if (root === null) throw new Error("the page has no #root to mount into");

createRoot(root).render(
  <StrictMode>
    <PuzzlesScreen />
  </StrictMode>,
);
