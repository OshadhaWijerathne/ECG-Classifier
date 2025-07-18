import "./App.css";
import "./Login.css";
import React, { useRef, useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";

const Login = () => {
  const [user, setUser] = useState("");
  const [pwd, setPwd] = useState("");

  const userRef = useRef();
  const errRef = useRef();

  const [errMsg, setErrMsg] = useState("");
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    userRef.current.focus(); // Here we simply set the focus to User input.
  }, []);

  useEffect(() => {
    setErrMsg("");
  }, [user, pwd]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log({ user, pwd });
    axios
      .post("http://localhost:4000/login", { user, pwd })
      .then((response) => {
        // Handle successful login
        console.log(
          response.data.status,
          response.data.message,
          response.data.data
        );
        if (response.data.status) {
          navigate("/useraccount");
        } else {
          alert(response.data.message);
          window.location.reload(false);
        }
      })
      .catch((error) => {
        // Handle login errors
        // console.error("Error logging in:", error);
        console.log("failed");
      });
  };

  return (
    <div className="login-page">
      <section>
        <p
          ref={errRef}
          className={errMsg ? "errmsg" : "offscreen"}
          aria-live="assertive"
        >
          {errMsg}
        </p>
        <form className="login-form" onSubmit={handleSubmit}>
          <h1 className="topic">Login</h1>
          <label className="field-name" htmlFor="username">
            Username:
          </label>
          <input
            className="typing-boxes"
            type="text"
            id="username"
            placeholder="Enter the Username"
            ref={userRef}
            autoComplete="off"
            onChange={(e) => setUser(e.target.value)}
            required
          />

          <label className="field-name" htmlFor="password">
            Password:
          </label>
          <input
            className="typing-boxes"
            type="password" // Type "password" will not support auto-complete anyway.
            id="password"
            placeholder="Enter your password"
            onChange={(e) => setPwd(e.target.value)}
            required
          />

          <button className="submit-button" type="submit">
            Log In
          </button>
          <p className="sign-up-link">
            Did not register yet?
            <br />
            <span className="line">
              <Link to="/register" className="link-btn">
                Register
              </Link>
            </span>
          </p>
        </form>
      </section>
    </div>
  );
};

export default Login;
