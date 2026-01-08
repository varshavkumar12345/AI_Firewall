// frontend/src/components/FirewallDemo.jsx
// Cybersecurity Dark Dashboard — Auto Mode Only (Visible Attack Waves)
// Frontend sends sequence history + current to backend /predict_seq every second
// Backend base: import.meta.env.VITE_API_BASE || http://127.0.0.1:8000
// Requires: recharts (npm i recharts)

import React, { useMemo, useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

// CORS helper for local development
const fetchWithCORS = async (url, options = {}) => {
  try {
    return await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
  } catch (err) {
    throw err;
  }
};

// local file (uploaded) — convert to URL for image usage if needed
const UPLOADED_IMAGE = "file:///mnt/data/012f61c3-55a1-4f20-8331-32654895e802.png";

const actionLabel = (a) => {
  if (a === 2) return { text: "BLOCK", color: "#ef4444" };
  if (a === 1) return { text: "RATE-LIMIT", color: "#f59e0b" };
  return { text: "ALLOW", color: "#10b981" };
};

// Utility: format time hh:mm:ss
const nowLabel = () => {
  const d = new Date();
  return d.toLocaleTimeString();
};



export default function FirewallDemo() {
  // --- UI & state ---
  const [latestDecision, setLatestDecision] = useState(null);
  const [history, setHistory] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);
  const [tsSeries, setTsSeries] = useState([]);
  const [running, setRunning] = useState(false);
  
  const timerRef = useRef(null);
  const historySeqRef = useRef([]);

  // Generate random IP
  const randomIp = () => `192.168.${Math.floor(Math.random() * 254)}.${Math.floor(Math.random() * 254)}`;

  // Generate realistic traffic data (baseline + normal variation)
  const generateTraffic = () => {
    const basePps = 100;  // Baseline: 100 pps
    const pps = basePps + Math.floor((Math.random() - 0.5) * 50);  // 75-125 pps
    
    const baseUnique = 8;
    const unique = baseUnique + Math.floor((Math.random() - 0.5) * 6);  // 5-11 unique IPs
    
    // Normal SYN ratio is around 5-8%
    const baseSyn = 0.06;
    const syn = Math.max(0.01, Math.min(0.99, baseSyn + (Math.random() - 0.5) * 0.03));
    
    return [pps, unique, Number(syn.toFixed(3))];
  };

  // Fetch real traffic data from backend
  const fetchTrafficData = async () => {
    try {
      const res = await fetch(`${API_BASE}/traffic`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return [data.pps, data.unique_ips, data.syn_ratio];
    } catch (err) {
      console.warn("Failed to fetch real traffic, using fallback:", err);
      // Fallback to generated data if backend unavailable
      return generateTraffic();
    }
  };

  // Fetch prediction from backend
  const fetchPrediction = async () => {
    try {
      // Get real traffic data from backend
      const current = await fetchTrafficData();
      const history_seq = [...historySeqRef.current, current].slice(-10);
      historySeqRef.current = history_seq;

      const payload = {
        history: history_seq,
        current: current,
        ip: randomIp(),
      };

      const res = await fetch(`${API_BASE}/predict_seq`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Server ${res.status}`);
      }

      const data = await res.json();

      const entry = {
        ts: nowLabel(),
        ip: payload.ip,
        current: current,
        action: data.action,
        suspicious: data.suspicious ?? 0,
      };

      setHistory((h) => [entry, ...h].slice(0, 200));
      setLatestDecision(entry);
      setTsSeries((s) => [...s, { time: nowLabel(), suspicious: data.suspicious ?? 0 }].slice(-80));
    } catch (err) {
      setErrorMsg(String(err));
    }
  };

  // Start/Stop polling
  const startPolling = () => {
    setRunning(true);
    setErrorMsg(null);
    timerRef.current = setInterval(fetchPrediction, 1000);
  };

  const stopPolling = () => {
    setRunning(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);



  // Chart data for bar counts
  const barData = useMemo(() => {
    const counts = { ALLOW: 0, "RATE-LIMIT": 0, BLOCK: 0, UNKNOWN: 0 };
    history.forEach((h) => {
      if (h.action === 0) counts.ALLOW++;
      else if (h.action === 1) counts["RATE-LIMIT"]++;
      else if (h.action === 2) counts.BLOCK++;
      else counts.UNKNOWN++;
    });
    return [
      { name: "ALLOW", count: counts.ALLOW, color: "#10b981" },
      { name: "RATE-LIMIT", count: counts["RATE-LIMIT"], color: "#f59e0b" },
      { name: "BLOCK", count: counts.BLOCK, color: "#ef4444" },
    ];
  }, [history]);

  // format small status card for latest
  const LatestCard = ({ latest }) => {
    const info = latest ? actionLabel(latest.action) : null;
    return (
      <div
        style={{
          background: "#061018",
          padding: 14,
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.03)",
        }}
      >
        <div style={{ color: "#94a3b8", fontSize: 13 }}>Latest Decision</div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 10 }}>
          <div
            style={{
              width: 70,
              height: 70,
              borderRadius: 12,
              background: info ? info.color : "#374151",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontWeight: 800,
              fontSize: 16,
            }}
          >
            {info ? info.text : "--"}
          </div>
          <div>
            <div style={{ color: "#e6eef6", fontWeight: 700 }}>{latest ? latest.ip : "—"}</div>
            <div style={{ color: "#94a3b8", fontSize: 12 }}>{latest ? latest.ts : ""}</div>
            <div style={{ color: "#94a3b8", marginTop: 6 }}>Wave: {latest ? latest.wave : "—"}</div>
            <div style={{ color: "#94a3b8", marginTop: 6 }}>
              Suspicious: {latest && latest.suspicious != null ? Number(latest.suspicious).toFixed(3) : "—"}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // === LAYOUT: make full-width container (changes from previous maxWidth center) ===
  return (
    <div style={{ width: "100vw", minHeight: "100vh", background: "linear-gradient(180deg,#041018 0%, #071624 100%)", padding: 24, boxSizing: "border-box" }}>
      {/* Full-width header bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, gap: 12 }}>
        <div style={{ flex: 1 }}>
          <h1 style={{ color: "#e6f0ff", margin: 0, fontSize: 32 }}>Autonomous AI Firewall — Live Dashboard</h1>
          <div style={{ color: "#94a3b8", marginTop: 6 }}>LSTM (suspicious) + XGBoost (decision)</div>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button
            onClick={startPolling}
            disabled={running}
            style={{
              background: "#06b6d4",
              color: "#022023",
              border: "none",
              padding: "10px 14px",
              borderRadius: 10,
              fontWeight: 800,
              cursor: running ? "not-allowed" : "pointer",
            }}
          >
            Start
          </button>
          <button
            onClick={stopPolling}
            disabled={!running}
            style={{
              background: "#374151",
              color: "#e6eef6",
              border: "none",
              padding: "10px 14px",
              borderRadius: 10,
              fontWeight: 700,
              cursor: !running ? "not-allowed" : "pointer",
            }}
          >
            Stop
          </button>
        </div>
      </div>

      {/* Main grid stretches full width */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 420px", gap: 20, alignItems: "start" }}>
        {/* Left: wide column */}
        <div>
          {/* Latest decision card */}
          <div style={{ display: "flex", gap: 14, marginBottom: 14 }}>
            <div style={{ width: 280 }}>
              <LatestCard latest={latestDecision} />
            </div>
          </div>

          {/* Suspicious trend chart */}
          <div style={{ background: "#07121a", padding: 14, borderRadius: 12, marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Suspicious Score (recent)</div>
              <div style={{ color: "#9ca3af", fontSize: 12 }}>{tsSeries.length} samples</div>
            </div>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tsSeries}>
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} stroke="#9ca3af" />
                  <Tooltip />
                  <Line type="monotone" dataKey="suspicious" stroke="#06b6d4" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Live decision feed */}
          <div style={{ background: "#07121a", padding: 14, borderRadius: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Live Decisions Feed</div>
              <div style={{ color: "#9ca3af", fontSize: 12 }}>{history.length} total</div>
            </div>

            <div style={{ marginTop: 10, maxHeight: "50vh", overflowY: "auto", paddingRight: 6 }}>
              {history.length === 0 && <div style={{ color: "#94a3b8" }}>No decisions yet.</div>}
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {history.map((h, i) => (
                  <li key={i} style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", background: "#0b1520", padding: 12, borderRadius: 10, marginBottom: 10 }}>
                    <div>
                      <div style={{ color: "#e6eef6", fontWeight: 800 }}>{h.ip} <span style={{ color: "#9ca3af", fontSize: 12, marginLeft: 8 }}>{h.ts}</span></div>
                      <div style={{ color: "#94a3b8", fontSize: 13, marginTop: 8 }}>pps: {h.current[0]}, ips: {h.current[1]}, syn: {h.current[2].toFixed(3)}</div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                      <div style={{ padding: "8px 12px", borderRadius: 8, background: actionLabel(h.action).color, color: "#fff", fontWeight: 900 }}>{actionLabel(h.action).text}</div>
                      <div style={{ color: "#94a3b8", fontSize: 12 }}>susp: {h.suspicious != null ? Number(h.suspicious).toFixed(3) : "—"}</div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Right: summary + chart (fixed column) */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ background: "#07121a", padding: 14, borderRadius: 12 }}>
            <div style={{ color: "#9ca3af", fontSize: 13 }}>Overview</div>
            <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ background: "#08121a", padding: 12, borderRadius: 8 }}>
                <div style={{ color: "#9ca3af", fontSize: 12 }}>Total Decisions</div>
                <div style={{ color: "#e6eef6", fontSize: 20, fontWeight: 800, marginTop: 8 }}>{history.length}</div>
              </div>
            </div>
          </div>

          <div style={{ background: "#07121a", padding: 14, borderRadius: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Decision Counts</div>
              <div style={{ color: "#9ca3af", fontSize: 12 }}>Last {history.length}</div>
            </div>

            <div style={{ width: "100%", height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData}>
                  <XAxis dataKey="name" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count">
                    {barData.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          
        </div>
      </div>

      {/* Footer / errors (full width) */}
      <div style={{ marginTop: 20 }}>
        {errorMsg && (
          <div style={{ background: "#fee2e2", color: "#991b1b", padding: 10, borderRadius: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>Error: {errorMsg}</div>
              <button onClick={() => setErrorMsg(null)} style={{ textDecoration: "underline", background: "transparent", border: "none" }}>Dismiss</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
