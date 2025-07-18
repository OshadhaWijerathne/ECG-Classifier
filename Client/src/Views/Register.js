import "./App.css";
import { Link, useNavigate } from "react-router-dom";
import React, { useRef, useState, useEffect } from "react";
import {
  faCheck,
  faTimes,
  faInfoCircle,
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import axios from "axios";

const USER_REGEX = /^[a-zA-Z][a-zA-Z0-9-_].{2,24}$/;
const PWD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%]).{8,24}$/;

const Register = (props) => {
  const userRef = useRef();
  const errRef = useRef();

  const [user, setUser] = useState("");
  const [validName, setValidName] = useState(false);
  const [userFocus, setUserFocus] = useState(false);

  const [pwd, setPwd] = useState("");
  const [validPwd, setValidPwd] = useState(false);
  const [pwdFocus, setPwdFocus] = useState(false);

  const [matchPwd, setMatchPwd] = useState("");
  const [validMatch, setValidMatch] = useState(false);
  const [matchFocus, setMatchFocus] = useState(false);

  const [errMsg, setErrMsg] = useState("");
  const [success, setSuccess] = useState(false);

  const navigate = useNavigate();

  useEffect(() => {
    userRef.current.focus(); // Here we simply set the focus to User input.
  }, []);

  useEffect(() => {
    // This is where we validate the user name.
    const result = USER_REGEX.test(user);
    console.log(user);
    console.log(result);
    setValidName(result);
  }, [user]); // Anytime this USER input changes, it will check the validation in that field.

  useEffect(() => {
    // This is where we validate the password.
    const result = PWD_REGEX.test(pwd);
    console.log(result);
    console.log(pwd);
    setValidPwd(result);
    const match = pwd === matchPwd; // By using both pwd and matchPwd within same EFFECT hook, a change in either of them will be checked with this single hook immediately.
    setValidMatch(match);
  }, [pwd, matchPwd]);

  useEffect(() => {
    setErrMsg("");
  }, [user, pwd, matchPwd]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    //If button was however enabled with a JS hack.
    const v1 = USER_REGEX.test(user);
    const v2 = PWD_REGEX.test(pwd);
    if (!v1 || !v2) {
      setErrMsg("Invalid Entry");
      return;
    }
    console.log(user, pwd);
    setSuccess(true);
    axios
      .post("http://localhost:4000/createUser", { user, pwd })
      .then((response) => {
        console.log(response.data.status, response.data.message);
        if (response.data.status) {
          alert("New user account created, Please login.");
          navigate("/login");
        } else {
          alert(response.data.message + " Please use a new Username.");
          window.location.reload(false);
          console.log("Page refreshed to enter new Reg. details.");
        }
      })
      .catch((error) => {
        // Handle registration errors
        console.log("failed with ", error);
      });
  };

  return (
    <section className="register-page">
      <p
        ref={errRef}
        className={errMsg ? "errmsg" : "offscreen"}
        aria-live="assertive"
      >
        {errMsg}
      </p>
      <form className="sign-up-form" onSubmit={handleSubmit}>
        <h1 className="topic">Register</h1>
        <label className="field-name" htmlFor="username">
          Username:
          <span className={validName ? "valid" : "hide"}>
            <FontAwesomeIcon
              icon={faCheck}
              style={{
                color: "green",
              }}
            />
          </span>
          <span className={validName || !user ? "hide" : "invalid"}>
            <FontAwesomeIcon
              icon={faTimes}
              style={{
                color: "red",
              }}
            />
          </span>
        </label>
        <input
          className="typing-boxes"
          type="text"
          id="username"
          placeholder="Enter your name"
          ref={userRef}
          autoComplete="off"
          onChange={(e) => setUser(e.target.value)}
          required
          aria-invalid={validName ? "false" : "true"}
          aria-describedby="uidnote" //This is where we allow the screen reader to see the Name assigning requirements.
          onFocus={() => {
            console.log("Username field focused");
            setUserFocus(true);
          }} //If user is on the input field set TRUE
          onBlur={() => {
            console.log("Username field blurred");
            setUserFocus(false);
          }} //Otherwise false.
        />

        <p
          id="uidnote"
          className={
            userFocus && user && !validName ? "instructions" : "offscreen"
          }
        >
          <FontAwesomeIcon icon={faInfoCircle} />
          &nbsp; 4 to 24 characters.
          <br />
          Must begin with a letter.
          <br />
          Letters,numbers,underscores,hyphens allowed.
        </p>

        <label className="field-name" htmlFor="password">
          Password:
          <span className={validPwd ? "valid" : "hide"}>
            <FontAwesomeIcon
              icon={faCheck}
              style={{
                color: "green",
              }}
            />
          </span>
          <span className={validPwd || !pwd ? "hide" : "invalid"}>
            <FontAwesomeIcon
              icon={faTimes}
              style={{
                color: "red",
              }}
            />
          </span>
        </label>
        <input
          className="typing-boxes"
          type="password" // Type "password" will not support auto-complete anyway.
          id="password"
          placeholder="Enter new password"
          onChange={(e) => setPwd(e.target.value)}
          required
          aria-invalid={validPwd ? "false" : "true"}
          aria-describedby="pwdnote" //This is where we allow the screen reader to see the Name assigning requirements.
          onFocus={() => {
            console.log("Password field focused");
            setPwdFocus(true);
          }} //If user is on the input field set TRUE
          onBlur={() => {
            console.log("Password field blurred");
            setPwdFocus(false);
          }} //Otherwise false.
        />
        <p
          id="pwdnote"
          className={pwdFocus && !validPwd ? "instructions" : "offscreen"}
        >
          <FontAwesomeIcon icon={faInfoCircle} />
          &nbsp; 8 to 24 characters.
          <br />
          Must include uppercase and lowercase letters, a number and a special
          character.
          <br />
          Allowed special characters:
          <span aria-label="exclaimation mark">!</span>
          <span aria-label="at symbol">@</span>
          <span aria-label="hashtag">#</span>
          <span aria-label="dollar sign">$</span>
          <span aria-label="percent">%</span>
        </p>

        <label className="field-name" htmlFor="confirm_pwd">
          Confirm Password:
          <span className={validMatch && matchPwd ? "valid" : "hide"}>
            <FontAwesomeIcon
              icon={faCheck}
              style={{
                color: "green",
              }}
            />
          </span>
          <span className={validMatch || !matchPwd ? "hide" : "invalid"}>
            <FontAwesomeIcon
              icon={faTimes}
              style={{
                color: "red",
              }}
            />
          </span>
        </label>
        <input
          className="typing-boxes"
          type="password" // Will not support auto-complete anyway.
          id="confirm_pwd"
          placeholder="Re-enter new password"
          onChange={(e) => setMatchPwd(e.target.value)}
          required
          aria-invalid={validMatch ? "false" : "true"}
          aria-describedby="confirmnote" //This is where we allow the screen reader to see the match password assigning requirements.
          onFocus={() => {
            console.log("Confirm-password field focused");
            setMatchFocus(true);
          }} //If user is on the input field set TRUE
          onBlur={() => {
            console.log("Confirm-password field blurred");
            setMatchFocus(false);
          }} //Otherwise false.
        />
        <p
          id="confirmnote"
          className={matchFocus && !validMatch ? "instructions" : "offscreen"}
        >
          <FontAwesomeIcon icon={faInfoCircle} />
          &nbsp; Must match the first password input field.
        </p>

        <button
          className="submit-button"
          type="submit"
          disabled={!validName || !validPwd || !validMatch ? true : false}
        >
          Sign Up
        </button>

        <p className="sign-in-link">
          Already registered?
          <br />
          <span className="line">
            <Link to="/login" className="link-btn">
              Log In
            </Link>
          </span>
        </p>
      </form>
    </section>
  );
};

export default Register;
