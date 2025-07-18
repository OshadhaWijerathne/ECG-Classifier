import React from "react";

export default function Navbar() {
  return (
    <nav className="navbar">
      <a href="./" className="dashboard-btn">
        Dashboard
      </a>
      <ul>
        <li>
          <a href="./About">About</a>
        </li>
        <li>
          <a href="./Login">Login</a>
        </li>
        <li>
          <a href="./Register">Register</a>
        </li>
      </ul>
    </nav>
  );
}
