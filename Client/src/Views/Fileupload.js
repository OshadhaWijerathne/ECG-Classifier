import React, { useRef, useState } from "react";
import "./App.css";
import { useNavigate } from "react-router-dom";

function FileUploader() {
  // Create references to the hidden file input elements
  const hiddenFileInput1 = useRef(null);
  const hiddenFileInput2 = useRef(null);

  const [matFile, setMatFile] = useState(null);
  const [headerFile, setHeaderFile] = useState(null);
  const navigate = useNavigate();

  // Programmatically click the hidden file input element when the Button component is clicked
  const handleClick = (inputRef) => () => {
    inputRef.current.click();
  };

  // Logic to handle the uploaded file goes here
  function handleFiles(file, fileIndex) {
    console.log("Handling file:", file, "with index:", fileIndex);
    if (fileIndex === 1) {
      console.log("Mat file uploaded.");
    } else {
      console.log("Header file uploaded.");
    }
  }

  // Call a function (passed as a prop from the parent component) to handle the user-selected files
  const handleChange = (event, fileIndex) => {
    const fileUploaded = event.target.files[0];
    handleFiles(fileUploaded, fileIndex);

    // Set the state with the selected file
    if (fileIndex === 1) {
      setMatFile(fileUploaded);
    } else if (fileIndex === 2) {
      setHeaderFile(fileUploaded);
    }
  };

  const handle_Submit = async (e) => {
    e.preventDefault();
    console.log("Two files submitted.");
    const formData = new FormData();

    if (matFile) {
      formData.append("matFile", matFile);
    }

    if (headerFile) {
      formData.append("headerFile", headerFile);
    }

    try {
      const response = await fetch("http://localhost:8000/userAccount", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json(); // Parse the JSON response
        // alert("Files Uploaded");
        navigate("/DataVisualization", { state: { data: data } });
      } else {
        console.error("Some error occurred.");
      }
    } catch (e) {
      console.error(e.message);
    }
  };

  return (
    <div className="buttons">
      <button
        className="file-upload-btn"
        onClick={handleClick(hiddenFileInput1)}
        disabled={matFile !== null}
      >
        Upload your .mat file here
      </button>
      <button
        className="file-upload-btn"
        onClick={handleClick(hiddenFileInput2)}
        disabled={headerFile !== null}
      >
        Upload your header file here
      </button>
      <input
        type="file"
        accept=".mat"
        multiple={false}
        required
        onChange={(event) => handleChange(event, 1)}
        ref={hiddenFileInput1}
        style={{ display: "none" }}
      />
      <input
        type="file"
        accept=".hea"
        multiple={false}
        required
        onChange={(event) => handleChange(event, 2)}
        ref={hiddenFileInput2}
        style={{ display: "none" }}
      />

      <button className="files-submit-btn" onClick={handle_Submit}>
        {" "}
        Submit Files
      </button>
    </div>
  );
}

export default FileUploader;
