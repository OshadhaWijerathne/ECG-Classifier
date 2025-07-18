import React, { PureComponent } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const data = [
  {
    name: "Page A",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page B",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page C",
    Generic_ECG_data_graph: 2,
  },
  {
    name: "Page D",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page E",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page F",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -2,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page A",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page B",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page C",
    Generic_ECG_data_graph: 2,
  },
  {
    name: "Page D",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page E",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page F",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -2,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page A",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page B",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page C",
    Generic_ECG_data_graph: 2,
  },
  {
    name: "Page D",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page E",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page F",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -2,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page A",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page B",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page C",
    Generic_ECG_data_graph: 2,
  },
  {
    name: "Page D",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page E",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page F",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -2,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page A",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page B",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page C",
    Generic_ECG_data_graph: 2,
  },
  {
    name: "Page D",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page E",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page F",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -2,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page A",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page B",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page C",
    Generic_ECG_data_graph: 2,
  },
  {
    name: "Page D",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page E",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page F",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -2,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: -1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 1,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0.5,
  },
  {
    name: "Page G",
    Generic_ECG_data_graph: 0,
  },
];

const ECGchart = () => {
  return (
    <LineChart
      className="linechart"
      width={600}
      height={230}
      data={data}
      margin={{
        top: 5,
        right: 40,
        left: 20,
        bottom: 20,
      }}
    >
      <CartesianGrid strokeDasharray="1 1" />
      <XAxis tick={false} />
      <YAxis />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey="Generic_ECG_data_graph" stroke="#FFA500" />
    </LineChart>
  );
};

export default ECGchart;
