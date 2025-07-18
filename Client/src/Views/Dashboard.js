import React from "react";
import DeathStats from "./DeathStats";
import ECGchart from "./ECGchart";

const Dashboard = () => {
  return (
    <div>
      <div className="line-chart">
        <ECGchart />
      </div>
      <div className="bar-chart">
        <DeathStats />
      </div>
    </div>
  );
};

export default Dashboard;
