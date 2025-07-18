import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./Views/Login";
import About from "./Views/About";
import Register from "./Views/Register";
import Dashboard from "./Views/Dashboard";
import Navbar from "./Views/Navbar";
import FileUploader from "./Views/Fileupload";
import DataVisualization from './Views/DataVisualization';


function App() {
  return (
    <div className="App">
      {/* Everything that need to be routed, must be placed inside opening and closing router tags. */}
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route exact path="/login" element={<Login />} />
          <Route path="/about" element={<About />} />
          <Route path="/register" element={<Register />} />
          <Route path="/useraccount" element={<FileUploader />} />
          <Route path="/DataVisualization" element={<DataVisualization />} />
        </Routes>
      </Router>
    </div>
  );
}
export default App;
