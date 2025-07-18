import React, { PureComponent } from "react";
import {
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const data = [
  {
    name: "2005",
    death_count: 653091,
  },
  {
    name: "2006",
    death_count: 631636,
  },
  {
    name: "2007",
    death_count: 616067,
  },
  {
    name: "2008",
    death_count: 616828,
  },
  {
    name: "2009",
    death_count: 599413,
  },
  {
    name: "2010",
    death_count: 597689,
  },
  {
    name: "2011",
    death_count: 596577,
  },
  {
    name: "2012",
    death_count: 599711,
  },
  {
    name: "2013",
    death_count: 611105,
  },
  {
    name: "2014",
    death_count: 614348,
  },
  {
    name: "2015",
    death_count: 633842,
  },
  {
    name: "2016",
    death_count: 635260,
  },
  {
    name: "2017",
    death_count: 647457,
  },
  {
    name: "2018",
    death_count: 655381,
  },
  {
    name: "2019",
    death_count: 659041,
  },
  {
    name: "2020",
    death_count: 686062,
  },
  {
    name: "2021",
    death_count: 874613,
  },
  {
    name: "2022",
    death_count: 880134,
  },
];

const DeathStats = () => {
  return (
    <ComposedChart
      className="barchart"
      width={600}
      height={250}
      data={data}
      margin={{
        top: 20,
        right: 20,
        bottom: 20,
        left: 20,
      }}
    >
      <CartesianGrid stroke="#FFFFFF" />
      <XAxis
        dataKey="name"
        scale="band"
        label={{
          value: "Year",
          position: "insideBottomRight",
          fontWeight: "bold",
          fill: "#FFFFFF",
        }}
        tick={{ fill: "#FFA500" }}
      />
      <YAxis
        label={{
          value: "Death Count",
          angle: -90,
          position: "insideLeft",
          fontWeight: "bold",
          fill: "#FFFFFF",
        }}
      />
      <Tooltip />
      <Legend />
      <Bar dataKey="death_count" barSize={20} fill="#FF8000" />
      <Line type="monotone" dataKey="death_count" stroke="#FFFFFF" />
    </ComposedChart>
  );
};

export default DeathStats;
