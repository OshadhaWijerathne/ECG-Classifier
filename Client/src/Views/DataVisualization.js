import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Plot from "react-plotly.js";

function DataVisualization() {
  const location = useLocation();
  const data = location.state.data;

  if (!data) {
    return <div>No data available</div>;
  }

  // Assuming mat_data is an array of data you want to plot
  const mat_data = data.Visualizing_data;

  // Create an array to hold the subplots
  const subplots = [];
  for (let lead = 0; lead < 12; lead++) {
    const slicedData = mat_data[lead].slice(0, 1000);
    subplots.push(
      <Plot
        key={lead}
        data={[
          {
            //y: mat_data[lead],
            y: slicedData, // Use sliced data
            type: "line",
          },
        ]}
        layout={{
          width: 600, // Set the width in pixels
          height: 300, // Set the height in pixels
          title: `Lead ${lead + 1}`,
          yaxis: {
            title: "Amplitude (mV)",
          },
          grid: {
            enabled: true,
          },
        }}
        config={{ responsive: true }}
      />
    );
  }

  const navigate = useNavigate();

  const NavigateToBegining = () => {
    navigate("/useraccount");
  };

  return (
    <div className="report">
      <h1 className="reportheading">Report</h1>
      <button className="uploadagain" onClick={NavigateToBegining}>
        Upload Again
      </button>
      <p className="demographicdata">Age: {data.Age}</p>
      <p className="demographicdata">Sex: {data.Sex}</p>
      <p className="predicted">You are suffering with: {data.predicted}</p>
      {/* <p>Real: {data.real[0]}</p> */}
      <div>{subplots}</div>
    </div>
  );
}

export default DataVisualization;
