"use client";

import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AnalysisResult } from "@/lib/types";

export function AngleChart({ result }: { result: AnalysisResult }) {
  const data = result.series.t.map((time, index) => ({ time, angle: result.series.angle[index] }));
  const joint = result.meta.exercise.vertex_name;

  return (
    <div className="chart-wrap" aria-label={`${joint} angle over time chart`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -12, bottom: 4 }}>
          <CartesianGrid stroke="#252a2f" vertical={false} />
          <XAxis dataKey="time" stroke="#727b84" tickFormatter={(value) => `${Number(value).toFixed(0)}s`} minTickGap={28} />
          <YAxis reversed stroke="#727b84" domain={["dataMin - 10", "dataMax + 10"]} tickFormatter={(value) => `${value}°`} />
          <Tooltip
            contentStyle={{ background: "#15191d", border: "1px solid #333a40", borderRadius: 12 }}
            labelFormatter={(value) => `${Number(value).toFixed(1)} seconds`}
            formatter={(value) => [`${Number(value).toFixed(1)}°`, `${joint} angle`]}
          />
          <ReferenceLine y={result.thresholds.full_rom} stroke="#80e84f" strokeDasharray="6 5" label={{ value: "Full ROM", fill: "#80e84f", fontSize: 11 }} />
          <ReferenceLine y={result.thresholds.up_enter} stroke="#6aa8ff" strokeDasharray="3 5" />
          <ReferenceLine y={result.thresholds.down_enter} stroke="#ff785a" strokeDasharray="3 5" />
          <Line type="monotone" dataKey="angle" stroke="#d9ff43" strokeWidth={3} dot={false} connectNulls={false} activeDot={{ r: 5, fill: "#d9ff43" }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
