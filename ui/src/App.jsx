import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:5000";

const MODES = [
  { key: "final", label: "Stabilized", hotkey: "0", color: "#00e5a0" },
  { key: "translation", label: "Translation", hotkey: "1", color: "#4da6ff" },
  { key: "rotation", label: "Rotation", hotkey: "2", color: "#ff8c42" },
  { key: "scaling", label: "Scaling", hotkey: "3", color: "#c77dff" },
  { key: "affine", label: "Affine", hotkey: "4", color: "#ff4d6d" },
  { key: "perspective", label: "Perspective", hotkey: "5", color: "#ffd166" },
  { key: "reflection", label: "Reflection", hotkey: "6", color: "#06d6a0" },
];

const fmt2 = (v) => (typeof v === "number" ? v.toFixed(2) : "—");
const fmt1 = (v) => (typeof v === "number" ? v.toFixed(1) : "—");
const fmt0 = (v) => (typeof v === "number" ? Math.round(v).toString() : "—");
const pct = (v) => (typeof v === "number" ? v.toFixed(1) + "%" : "—");

// ── colour palette ────────────────────────────────────────────────────────────
// gold  #ffd166   green #00e5a0   blue #4da6ff   muted #aaa   dim #777
// Used consistently: section titles → gold, values → accent colour, labels → #aaa

function DualSparkline({
  data1,
  data2,
  c1 = "#ff4d4d",
  c2 = "#00e5a0",
  height = 56,
}) {
  if (!data1 || data1.length < 2)
    return (
      <div
        style={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#777",
          fontSize: 11,
        }}
      >
        no data yet
      </div>
    );
  const W = 300,
    H = height;
  const maxV = Math.max(...data1, ...(data2 || []), 0.001);
  const pts = (arr) =>
    arr
      .map((v, i) => {
        const x = 2 + (i / (arr.length - 1)) * (W - 4);
        const y = H - 4 - (v / maxV) * (H - 8);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  return (
    <svg
      width="100%"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      style={{ display: "block" }}
    >
      {data1.length > 1 && (
        <polyline
          points={pts(data1)}
          fill="none"
          stroke={c1}
          strokeWidth="1.5"
          strokeLinejoin="round"
          opacity="0.9"
        />
      )}
      {data2 && data2.length > 1 && (
        <polyline
          points={pts(data2)}
          fill="none"
          stroke={c2}
          strokeWidth="1.5"
          strokeLinejoin="round"
          opacity="0.9"
        />
      )}
    </svg>
  );
}

function HeatBar({ data, height = 44 }) {
  if (!data || !data.length) return null;
  const maxV = Math.max(...data, 0.001);
  const bars = data.slice(-60);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height }}>
      {bars.map((v, i) => {
        const norm = v / maxV;
        const h = Math.max(3, norm * height);
        return (
          <div
            key={i}
            style={{
              flex: 1,
              height: h,
              background: `rgb(${Math.round(255 * norm)},${Math.round(200 * (1 - norm))},30)`,
              borderRadius: 1,
            }}
          />
        );
      })}
    </div>
  );
}

function Stat({ label, value, unit = "", color = "#00e5a0" }) {
  return (
    <div
      style={{
        padding: "10px 12px",
        background: "#0d0d0d",
        border: "1px solid #2a2a2a",
        borderRadius: 6,
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: "#ffd166",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          marginBottom: 4,
          fontFamily: "monospace",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 18,
          fontWeight: 600,
          color,
          fontFamily: "monospace",
        }}
      >
        {value}
        <span style={{ fontSize: 11, color: "#777", marginLeft: 3 }}>
          {unit}
        </span>
      </div>
    </div>
  );
}

function ProgressBar({ value, max, color = "#00e5a0" }) {
  const p = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div
      style={{
        background: "#111",
        borderRadius: 3,
        height: 5,
        overflow: "hidden",
        border: "1px solid #2a2a2a",
      }}
    >
      <div
        style={{
          width: `${p}%`,
          height: "100%",
          background: color,
          transition: "width 0.08s",
          borderRadius: 3,
        }}
      />
    </div>
  );
}

function Panel({ title, children, style }) {
  return (
    <div
      style={{
        background: "#0d0d0d",
        border: "1px solid #2a2a2a",
        borderRadius: 10,
        padding: 16,
        ...style,
      }}
    >
      {title && (
        <div
          style={{
            fontSize: 10,
            color: "#ffd166",
            fontFamily: "monospace",
            letterSpacing: "0.12em",
            marginBottom: 12,
            textTransform: "uppercase",
          }}
        >
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

function LegendDot({ color, label }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        fontSize: 10,
        color: "#aaa",
      }}
    >
      <span
        style={{
          width: 14,
          height: 2,
          background: color,
          display: "inline-block",
          borderRadius: 1,
        }}
      />
      {label}
    </div>
  );
}

function ModeBar({ mode, onChange, disabled }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {MODES.map((m) => (
        <button
          key={m.key}
          onClick={() => !disabled && onChange(m.key)}
          style={{
            padding: "5px 11px",
            fontSize: 11,
            fontFamily: "monospace",
            background: mode === m.key ? m.color + "22" : "#111",
            color: mode === m.key ? m.color : "#aaa",
            border: `1px solid ${mode === m.key ? m.color + "88" : "#333"}`,
            borderRadius: 5,
            cursor: disabled ? "not-allowed" : "pointer",
            transition: "all 0.12s",
          }}
        >
          [{m.hotkey}] {m.label}
        </button>
      ))}
    </div>
  );
}

// ── VideoFeed with fullscreen zoom ────────────────────────────────────────────
function VideoFeed({ label, accent, running, streamPath }) {
  const [imgError, setImgError] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    setImgError(false);
  }, [running, streamPath]);

  // Close fullscreen on Escape key
  useEffect(() => {
    if (!fullscreen) return;
    const handler = (e) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [fullscreen]);

  const streamSrc = `${API}${streamPath}`;

  return (
    <>
      {/* ── Fullscreen overlay modal ── */}
      {fullscreen && (
        <div
          onClick={() => setFullscreen(false)}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 9999,
            background: "rgba(0,0,0,0.93)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {/* Top bar — stopPropagation so clicks here don't close the modal */}
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "10px 20px",
              boxSizing: "border-box",
              flexShrink: 0,
            }}
          >
            <span
              style={{
                fontSize: 12,
                color: accent,
                fontFamily: "monospace",
                letterSpacing: "0.08em",
              }}
            >
              {label}
            </span>
            {running && (
              <span
                style={{
                  fontSize: 11,
                  color: accent + "cc",
                  fontFamily: "monospace",
                }}
              >
                ● LIVE
              </span>
            )}
            <button
              onClick={() => setFullscreen(false)}
              style={{
                fontSize: 12,
                fontFamily: "monospace",
                padding: "5px 16px",
                background: accent + "22",
                color: accent,
                border: `1px solid ${accent}66`,
                borderRadius: 5,
                cursor: "pointer",
              }}
            >
              ⊡ EXIT FULLSCREEN [Esc]
            </button>
          </div>

          {/* Image area — fills all remaining space, image is letterboxed inside */}
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              flex: 1,
              width: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "0 20px 20px",
              boxSizing: "border-box",
              minHeight: 0, // critical: lets flex child shrink below content size
            }}
          >
            {running && !imgError ? (
              <img
                src={streamSrc}
                alt={label}
                onError={() => setImgError(true)}
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  width: "auto",
                  height: "auto",
                  objectFit: "contain",
                  display: "block",
                  border: `1px solid ${accent}44`,
                  borderRadius: 6,
                }}
              />
            ) : (
              <div
                style={{
                  fontSize: 13,
                  color: "#aaa",
                  fontFamily: "monospace",
                  textAlign: "center",
                }}
              >
                {running && imgError ? (
                  <>
                    <span style={{ color: "#ffaa44" }}>
                      ⚠ stream unavailable
                    </span>
                    <br />
                    <span style={{ fontSize: 11, color: "#777" }}>
                      check /video_feed/* routes in api.py
                    </span>
                  </>
                ) : (
                  <span style={{ color: "#777" }}>waiting for source</span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Normal inline feed ── */}
      <div
        style={{ flex: 1, display: "flex", flexDirection: "column", gap: 4 }}
      >
        {/* label + fullscreen button row */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span
            style={{
              fontSize: 10,
              color: accent,
              fontFamily: "monospace",
              letterSpacing: "0.08em",
            }}
          >
            {label}
          </span>
          <button
            onClick={() => setFullscreen(true)}
            style={{
              fontSize: 10,
              fontFamily: "monospace",
              padding: "2px 8px",
              background: "#111",
              color: "#777",
              border: "1px solid #333",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            ⊞ fullscreen
          </button>
        </div>

        {/* fixed-aspect container */}
        <div
          style={{
            position: "relative",
            overflow: "hidden",
            aspectRatio: "16/9",
            background: "#050505",
            border: `1px solid ${running ? accent + "55" : "#2a2a2a"}`,
            borderRadius: 6,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
              backgroundImage: `repeating-linear-gradient(0deg,transparent,transparent 20px,${accent}08 20px,${accent}08 21px),
                              repeating-linear-gradient(90deg,transparent,transparent 20px,${accent}08 20px,${accent}08 21px)`,
            }}
          />

          {running && (
            <div
              style={{
                position: "absolute",
                bottom: 8,
                right: 10,
                fontSize: 10,
                color: accent + "cc",
                fontFamily: "monospace",
                zIndex: 2,
              }}
            >
              ● LIVE
            </div>
          )}

          {running && !imgError ? (
            <img
              src={streamSrc}
              alt={label}
              onError={() => setImgError(true)}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
                display: "block",
                zIndex: 1,
              }}
            />
          ) : (
            <div
              style={{
                fontSize: 12,
                color: "#aaa",
                fontFamily: "monospace",
                zIndex: 1,
                textAlign: "center",
                padding: 8,
              }}
            >
              {running && imgError ? (
                <>
                  <span style={{ color: "#ffaa44" }}>⚠ stream unavailable</span>
                  <br />
                  <span style={{ fontSize: 10, color: "#777" }}>
                    check /video_feed/* routes in api.py
                  </span>
                </>
              ) : (
                <span style={{ color: "#777" }}>waiting for source</span>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

function HUD({ metrics }) {
  const m = metrics;
  const impColor = m && m.improvement > 0 ? "#00e5a0" : "#ff4d4d";
  return (
    <div
      style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}
    >
      <Stat label="dX" value={m ? fmt1(m.dx) : "—"} unit="px" color="#4da6ff" />
      <Stat label="dY" value={m ? fmt1(m.dy) : "—"} unit="px" color="#4da6ff" />
      <Stat
        label="Features"
        value={m ? fmt0(m.features) : "—"}
        color="#ffd166"
      />
      <Stat
        label="Raw Disp."
        value={m ? fmt2(m.raw_disp) : "—"}
        unit="px"
        color="#ff6b6b"
      />
      <Stat
        label="Stab Disp."
        value={m ? fmt2(m.stab_disp) : "—"}
        unit="px"
        color="#00e5a0"
      />
      <Stat label="FPS" value={m ? fmt0(m.fps) : "—"} color="#c77dff" />
      <Stat
        label="ROI Raw"
        value={m ? fmt2(m.roi_raw) : "—"}
        unit="px"
        color="#ff8c42"
      />
      <Stat
        label="ROI Stab"
        value={m ? fmt2(m.roi_stab) : "—"}
        unit="px"
        color="#06d6a0"
      />
      <Stat
        label="ROI Improv."
        value={m ? pct(m.improvement) : "—"}
        color={impColor}
      />
    </div>
  );
}

function Graphs({ H }) {
  const panels = [
    {
      title: "Displacement Magnitude",
      d1: H.raw,
      d2: H.stab,
      c1: "#ff4d4d",
      c2: "#00e5a0",
    },
    {
      title: "ROI Displacement",
      d1: H.roi_raw,
      d2: H.roi_stab,
      c1: "#ff8c42",
      c2: "#06d6a0",
    },
    {
      title: "X Component (abs)",
      d1: H.dx,
      d2: H.stab,
      c1: "#c77dff",
      c2: "#4da6ff",
    },
    {
      title: "Y Component (abs)",
      d1: H.dy,
      d2: H.stab,
      c1: "#ffd166",
      c2: "#00e5a0",
    },
    { title: "FPS Timeline", d1: H.fps, d2: [], c1: "#4da6ff", c2: "#555" },
    { title: "Motion Intensity", heatmap: true },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {panels.map(({ title, d1, d2, c1, c2, heatmap }) => (
        <Panel key={title} title={title}>
          {heatmap ? (
            <HeatBar data={H.raw} height={56} />
          ) : (
            <>
              <DualSparkline
                data1={d1}
                data2={d2}
                c1={c1}
                c2={c2}
                height={60}
              />
              {d2 && d2.length > 0 && (
                <div style={{ display: "flex", gap: 14, marginTop: 6 }}>
                  <LegendDot
                    color={c1}
                    label={title.includes("FPS") ? "fps" : "raw"}
                  />
                  <LegendDot color={c2} label="stab" />
                </div>
              )}
            </>
          )}
        </Panel>
      ))}
    </div>
  );
}

function StatsPanel({ sum }) {
  if (!sum)
    return (
      <div style={{ color: "#aaa", fontSize: 12, fontFamily: "monospace" }}>
        No summary yet — finish processing a video first.
      </div>
    );
  const d = sum.detailed || {};
  const rows = [
    ["Raw Mean", fmt2(d.raw_mean), "px", "#ff6b6b"],
    ["Raw Std Dev", fmt2(d.raw_std), "px", "#ff8c42"],
    ["Raw Max", fmt2(d.raw_max), "px", "#ff4d4d"],
    ["Stab Mean", fmt2(d.stab_mean), "px", "#00e5a0"],
    ["Stab Std Dev", fmt2(d.stab_std), "px", "#06d6a0"],
    ["Stab Max", fmt2(d.stab_max), "px", "#4da6ff"],
    ["Raw dX Mean", fmt2(d.raw_dx_mean), "px", "#c77dff"],
    ["Raw dY Mean", fmt2(d.raw_dy_mean), "px", "#c77dff"],
    ["Stab dX Mean", fmt2(d.stab_dx_mean), "px", "#ffd166"],
    ["Stab dY Mean", fmt2(d.stab_dy_mean), "px", "#ffd166"],
    [
      "Overall Improvement",
      pct(sum.improvement),
      "",
      sum.improvement > 0 ? "#00e5a0" : "#ff4d4d",
    ],
    ["Frames Processed", sum.frames, "", "#ccc"],
  ];
  return (
    <div>
      {rows.map(([label, val, unit, color]) => (
        <div
          key={label}
          style={{
            display: "flex",
            justifyContent: "space-between",
            padding: "7px 0",
            borderBottom: "1px solid #1a1a1a",
          }}
        >
          <span
            style={{ fontSize: 11, color: "#aaa", fontFamily: "monospace" }}
          >
            {label}
          </span>
          <span
            style={{
              fontSize: 12,
              color,
              fontFamily: "monospace",
              fontWeight: 600,
            }}
          >
            {val}{" "}
            <span style={{ color: "#777", fontWeight: 400, fontSize: 10 }}>
              {unit}
            </span>
          </span>
        </div>
      ))}
    </div>
  );
}

function BatchTable({ summaries }) {
  const keys = Object.keys(summaries);
  if (!keys.length)
    return (
      <div style={{ color: "#aaa", fontSize: 12, fontFamily: "monospace" }}>
        No videos processed yet.
      </div>
    );
  return (
    <table
      style={{
        width: "100%",
        borderCollapse: "collapse",
        fontSize: 12,
        fontFamily: "monospace",
      }}
    >
      <thead>
        <tr style={{ borderBottom: "1px solid #2a2a2a" }}>
          {[
            "Video",
            "Frames",
            "Raw (px)",
            "Stab (px)",
            "ROI Raw",
            "ROI Stab",
            "Improvement",
          ].map((h) => (
            <th
              key={h}
              style={{
                padding: "7px 10px",
                textAlign: "left",
                color: "#ffd166",
                fontSize: 10,
                fontWeight: 500,
                letterSpacing: "0.07em",
              }}
            >
              {h.toUpperCase()}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => {
          const s = summaries[k];
          const imp = s.improvement;
          return (
            <tr key={k} style={{ borderBottom: "1px solid #1a1a1a" }}>
              <td style={{ padding: "8px 10px", color: "#ddd" }}>{k}</td>
              <td style={{ padding: "8px 10px", color: "#aaa" }}>{s.frames}</td>
              <td style={{ padding: "8px 10px", color: "#ff6b6b" }}>
                {fmt2(s.raw_disp)}
              </td>
              <td style={{ padding: "8px 10px", color: "#00e5a0" }}>
                {fmt2(s.stab_disp)}
              </td>
              <td style={{ padding: "8px 10px", color: "#ff8c42" }}>
                {fmt2(s.roi_raw)}
              </td>
              <td style={{ padding: "8px 10px", color: "#06d6a0" }}>
                {fmt2(s.roi_stab)}
              </td>
              <td
                style={{
                  padding: "8px 10px",
                  color: imp > 0 ? "#00e5a0" : "#ff4d4d",
                  fontWeight: 600,
                }}
              >
                {pct(imp)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ page, setPage, mode, changeMode, apiOk, running, source }) {
  return (
    <div
      style={{
        width: 210,
        flexShrink: 0,
        background: "#090909",
        borderRight: "1px solid #1e1e1e",
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
      }}
    >
      <div
        style={{ padding: "18px 16px 12px", borderBottom: "1px solid #1e1e1e" }}
      >
        <div
          style={{
            fontSize: 17,
            fontWeight: 700,
            color: "#fff",
            fontFamily: "monospace",
            letterSpacing: "0.06em",
          }}
        >
          FRAME<span style={{ color: "#00e5a0" }}>LOCK</span>
        </div>
        <div
          style={{
            fontSize: 7,
            color: "#ffd166",
            marginTop: 2,
            letterSpacing: "0.15em",
          }}
        >
          SURGICAL VIDEO MOTION STABILIZER
        </div>
      </div>

      <div style={{ padding: "10px 8px", flex: 1 }}>
        {[
          { id: "home", icon: "⬡", label: "Dashboard" },
          { id: "dataset", icon: "▦", label: "Dataset Videos" },
          { id: "camera", icon: "◎", label: "Live Camera" },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => setPage(item.id)}
            style={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "9px 12px",
              marginBottom: 3,
              background: page === item.id ? "#00e5a011" : "transparent",
              color: page === item.id ? "#00e5a0" : "#ccc",
              border: `1px solid ${page === item.id ? "#00e5a022" : "transparent"}`,
              borderRadius: 7,
              cursor: "pointer",
              fontSize: 13,
              fontFamily: "monospace",
              transition: "all 0.12s",
              textAlign: "left",
            }}
          >
            <span>{item.icon}</span>
            {item.label}
          </button>
        ))}

        <div style={{ height: 1, background: "#1e1e1e", margin: "10px 4px" }} />
        <div
          style={{
            fontSize: 9,
            color: "#ffd166",
            letterSpacing: "0.12em",
            padding: "4px 12px 6px",
          }}
        >
          TRANSFORM MODE
        </div>

        {MODES.map((m) => (
          <button
            key={m.key}
            onClick={() => changeMode(m.key)}
            style={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "6px 12px",
              background: mode === m.key ? m.color + "14" : "transparent",
              color: mode === m.key ? m.color : "#aaa",
              border: "none",
              borderRadius: 5,
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "monospace",
              transition: "all 0.1s",
            }}
          >
            <span>{m.label}</span>
            <span style={{ opacity: 0.6, fontSize: 10, color: "#777" }}>
              [{m.hotkey}]
            </span>
          </button>
        ))}
      </div>

      <div style={{ padding: "12px 16px", borderTop: "1px solid #1e1e1e" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 7,
            fontSize: 11,
            fontFamily: "monospace",
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              flexShrink: 0,
              background: apiOk ? (running ? "#00e5a0" : "#2a4a3a") : "#4a2a2a",
              boxShadow: running ? "0 0 6px #00e5a0" : "none",
            }}
          />
          <span
            style={{
              color: apiOk ? (running ? "#00e5a0" : "#aaa") : "#ff4d4d",
            }}
          >
            {apiOk ? (running ? "PROCESSING" : "READY") : "API OFFLINE"}
          </span>
        </div>
        {source && (
          <div
            style={{
              fontSize: 10,
              color: "#777",
              fontFamily: "monospace",
              marginTop: 4,
              wordBreak: "break-all",
            }}
          >
            {source}
          </div>
        )}
      </div>
    </div>
  );
}

// ── HomePage ──────────────────────────────────────────────────────────────────
function HomePage({ setPage, videos, summaries, apiOk }) {
  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <div
          style={{
            fontSize: 22,
            color: "#fff",
            fontWeight: 700,
            fontFamily: "monospace",
          }}
        >
          Dashboard
        </div>
        <div style={{ fontSize: 12, color: "#aaa", marginTop: 4 }}>
          Video stabilization pipeline — choose a source to begin
        </div>
      </div>

      {!apiOk && (
        <div
          style={{
            background: "#1a0a0a",
            border: "1px solid #ff4d4d44",
            borderRadius: 8,
            padding: 16,
            marginBottom: 20,
          }}
        >
          <div
            style={{ fontSize: 12, color: "#ff6b6b", fontFamily: "monospace" }}
          >
            ⚠ API offline — start api.py first:
          </div>
          <code
            style={{
              fontSize: 11,
              color: "#ff9999",
              display: "block",
              marginTop: 6,
            }}
          >
            cd src && python api.py
          </code>
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
          maxWidth: 720,
          marginBottom: 28,
        }}
      >
        {[
          {
            id: "dataset",
            icon: "▦",
            title: "Dataset Videos",
            color: "#4da6ff",
            desc: "Process .mp4 files from data/input/. View displacement plots, motion heatmaps, ROI tracking, and per-video summaries.",
            points: [
              `${videos.length} video${videos.length !== 1 ? "s" : ""} in data/input/`,
              "Batch results table",
              "Full metrics suite",
            ],
          },
          {
            id: "camera",
            icon: "◎",
            title: "Live Camera",
            color: "#00e5a0",
            desc: "Stabilize your webcam feed in real time. Side-by-side view with live HUD, displacement graphs, and motion heatmap.",
            points: [
              "Real-time optical flow",
              "Side-by-side preview",
              "Live HUD overlay",
            ],
          },
        ].map((card) => (
          <div
            key={card.id}
            onClick={() => setPage(card.id)}
            style={{
              background: "#0d0d0d",
              border: "1px solid #2a2a2a",
              borderRadius: 10,
              padding: 22,
              cursor: "pointer",
              transition: "border-color 0.15s",
            }}
          >
            <div style={{ fontSize: 26, color: card.color, marginBottom: 10 }}>
              {card.icon}
            </div>
            <div
              style={{
                fontSize: 15,
                color: "#fff",
                fontWeight: 600,
                fontFamily: "monospace",
                marginBottom: 8,
              }}
            >
              {card.title}
            </div>
            <div
              style={{
                fontSize: 12,
                color: "#aaa",
                lineHeight: 1.7,
                marginBottom: 14,
              }}
            >
              {card.desc}
            </div>
            {card.points.map((p) => (
              <div
                key={p}
                style={{
                  fontSize: 11,
                  color: "#aaa",
                  fontFamily: "monospace",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  marginBottom: 4,
                }}
              >
                <span style={{ color: card.color, fontSize: 7 }}>◆</span>
                {p}
              </div>
            ))}
          </div>
        ))}
      </div>

      {Object.keys(summaries).length > 0 && (
        <Panel title="Recent Results">
          <BatchTable summaries={summaries} />
        </Panel>
      )}
    </div>
  );
}

// ── DatasetPage ───────────────────────────────────────────────────────────────
function DatasetPage({
  videos,
  running,
  source,
  metrics,
  summaries,
  mode,
  tab,
  setTab,
  H,
  startSource,
  stopProcessing,
  changeMode,
  fetchVideos,
}) {
  const currentSum = source ? summaries[source] : null;
  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 20,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 20,
              color: "#fff",
              fontWeight: 700,
              fontFamily: "monospace",
            }}
          >
            Dataset Videos
          </div>
          <div style={{ fontSize: 11, color: "#aaa", marginTop: 3 }}>
            {videos.length} file{videos.length !== 1 ? "s" : ""} in data/input/
          </div>
        </div>
        {running && (
          <button
            onClick={stopProcessing}
            style={{
              padding: "7px 16px",
              background: "#ff4d4d22",
              color: "#ff6b6b",
              border: "1px solid #ff4d4d44",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontFamily: "monospace",
            }}
          >
            ■ STOP
          </button>
        )}
      </div>

      {videos.length === 0 && (
        <div
          style={{
            background: "#0a0a0a",
            border: "1px solid #2a2a2a",
            borderRadius: 8,
            padding: 18,
            marginBottom: 20,
          }}
        >
          <div style={{ fontSize: 12, color: "#ccc", fontFamily: "monospace" }}>
            No videos found in{" "}
            <code style={{ color: "#ffd166" }}>data/input/</code>
          </div>
          <div style={{ fontSize: 11, color: "#aaa", marginTop: 6 }}>
            Add .mp4 / .avi / .mov files to the input folder, then refresh.
          </div>
          <button
            onClick={fetchVideos}
            style={{
              marginTop: 10,
              padding: "5px 14px",
              background: "#111",
              color: "#aaa",
              border: "1px solid #333",
              borderRadius: 5,
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "monospace",
            }}
          >
            ↻ Refresh
          </button>
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
          marginBottom: 20,
        }}
      >
        {videos.map((v) => {
          const isActive = running && source === v.name;
          const sum = summaries[v.name];
          return (
            <div
              key={v.name}
              onClick={() => !running && startSource(v.name)}
              style={{
                background: "#0d0d0d",
                border: `1px solid ${isActive ? "#00e5a044" : sum ? "#1e3a2a" : "#2a2a2a"}`,
                borderRadius: 8,
                padding: 16,
                cursor: running ? "not-allowed" : "pointer",
                transition: "all 0.12s",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  marginBottom: 10,
                }}
              >
                <div
                  style={{
                    fontSize: 12,
                    color: "#fff",
                    fontFamily: "monospace",
                    wordBreak: "break-all",
                  }}
                >
                  {v.name}
                </div>
                <span
                  style={{
                    fontSize: 9,
                    padding: "2px 7px",
                    borderRadius: 4,
                    fontFamily: "monospace",
                    flexShrink: 0,
                    marginLeft: 8,
                    background: isActive
                      ? "#00e5a022"
                      : sum
                        ? "#00e5a011"
                        : "#1a1a1a",
                    color: isActive ? "#00e5a0" : sum ? "#00c87a" : "#aaa",
                    border: `1px solid ${isActive ? "#00e5a044" : sum ? "#00e5a022" : "#333"}`,
                  }}
                >
                  {isActive ? "RUNNING" : sum ? "DONE" : "READY"}
                </span>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(4,1fr)",
                  gap: 6,
                  marginBottom: 10,
                }}
              >
                {[
                  ["frames", v.frames],
                  ["fps", v.fps],
                  ["dur", v.duration + "s"],
                  ["size", v.size_mb + "MB"],
                ].map(([l, val]) => (
                  <div key={l} style={{ fontSize: 12, color: "#ffd166" }}>
                    {l.charAt(0).toUpperCase() + l.slice(1)}
                    <br />
                    <span style={{ color: "#ccc" }}>{val}</span>
                  </div>
                ))}
              </div>

              {isActive && metrics && (
                <>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: 10,
                      color: "#aaa",
                      marginBottom: 4,
                    }}
                  >
                    <span>
                      frame{" "}
                      <span style={{ color: "#ffd166" }}>{metrics.frame}</span>{" "}
                      / {metrics.total || "?"}
                    </span>
                    <span>
                      <span style={{ color: "#00e5a0" }}>
                        {fmt0(metrics.fps)}
                      </span>{" "}
                      fps ·{" "}
                      <span style={{ color: "#4da6ff" }}>
                        {fmt0(metrics.features)}
                      </span>{" "}
                      pts
                    </span>
                  </div>
                  <ProgressBar value={metrics.frame} max={metrics.total || 1} />
                  <div style={{ marginTop: 10 }}>
                    <DualSparkline data1={H.raw} data2={H.stab} height={40} />
                  </div>
                </>
              )}

              {sum && !isActive && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: 6,
                  }}
                >
                  {[
                    ["raw", fmt2(sum.raw_disp) + "px", "#ff6b6b"],
                    ["stab", fmt2(sum.stab_disp) + "px", "#00e5a0"],
                    [
                      "impr.",
                      pct(sum.improvement),
                      sum.improvement > 0 ? "#00e5a0" : "#ff4d4d",
                    ],
                  ].map(([l, val, c]) => (
                    <div key={l} style={{ fontSize: 12, color: "#ffd166" }}>
                      {l.charAt(0).toUpperCase() + l.slice(1)}
                      <br />
                      <span style={{ color: c, fontFamily: "monospace" }}>
                        {val}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {!sum && !isActive && (
                <div
                  style={{
                    fontSize: 11,
                    color: "#777",
                    fontFamily: "monospace",
                    textAlign: "center",
                  }}
                >
                  click to process
                </div>
              )}
            </div>
          );
        })}
      </div>

      {(running || Object.keys(summaries).length > 0) && (
        <>
          <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
            {[
              ["live", "LIVE VIEW"],
              ["graphs", "GRAPHS"],
              ["stats", "STATS"],
              ["batch", "BATCH"],
            ].map(([id, label]) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                style={{
                  padding: "5px 14px",
                  fontSize: 11,
                  fontFamily: "monospace",
                  background: tab === id ? "#00e5a022" : "#111",
                  color: tab === id ? "#00e5a0" : "#aaa",
                  border: `1px solid ${tab === id ? "#00e5a044" : "#2a2a2a"}`,
                  borderRadius: 5,
                  cursor: "pointer",
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "live" && (
            <Panel>
              <div
                style={{
                  fontSize: 10,
                  color: "#aaa",
                  fontFamily: "monospace",
                  marginBottom: 12,
                }}
              >
                <span style={{ color: "#ffd166" }}>SIDE-BY-SIDE</span> —{" "}
                <span style={{ color: "#ccc" }}>{source || "—"}</span> — MODE:{" "}
                <span
                  style={{ color: MODES.find((m2) => m2.key === mode)?.color }}
                >
                  {mode.toUpperCase()}
                </span>
              </div>
              <div style={{ display: "flex", gap: 14, marginBottom: 14 }}>
                <VideoFeed
                  label="● ORIGINAL"
                  accent="#ff4d4d"
                  running={running}
                  streamPath="/video_feed/raw"
                />
                <VideoFeed
                  label="◆ STABILIZED"
                  accent="#00e5a0"
                  running={running}
                  streamPath="/video_feed/stabilized"
                />
              </div>
              {metrics && (
                <ProgressBar value={metrics.frame} max={metrics.total || 1} />
              )}
              <div style={{ margin: "14px 0" }}>
                <HUD metrics={metrics} />
              </div>
              <div style={{ marginTop: 14 }}>
                <div
                  style={{
                    fontSize: 10,
                    color: "#ffd166",
                    fontFamily: "monospace",
                    marginBottom: 8,
                    letterSpacing: "0.1em",
                  }}
                >
                  TRANSFORM MODE
                </div>
                <ModeBar mode={mode} onChange={changeMode} />
              </div>
            </Panel>
          )}
          {tab === "graphs" && <Graphs H={H} />}
          {tab === "stats" && (
            <Panel title="Detailed Statistics">
              <StatsPanel sum={currentSum} />
            </Panel>
          )}
          {tab === "batch" && (
            <Panel title="Batch Results">
              <BatchTable summaries={summaries} />
            </Panel>
          )}
        </>
      )}
    </div>
  );
}

// ── CameraPage ────────────────────────────────────────────────────────────────
function CameraPage({
  running,
  source,
  metrics,
  mode,
  H,
  startSource,
  stopProcessing,
  changeMode,
}) {
  const camRunning = running && source === "camera";
  const [camTab, setCamTab] = useState("live");

  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 20,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 20,
              color: "#fff",
              fontWeight: 700,
              fontFamily: "monospace",
            }}
          >
            Live Camera
          </div>
          <div style={{ fontSize: 11, color: "#aaa", marginTop: 3 }}>
            Real-time optical flow stabilization via webcam
          </div>
        </div>
        {!camRunning ? (
          <button
            onClick={() => startSource("camera")}
            style={{
              padding: "8px 20px",
              background: "#00e5a022",
              color: "#00e5a0",
              border: "1px solid #00e5a044",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontFamily: "monospace",
              letterSpacing: "0.05em",
            }}
          >
            ● START CAMERA
          </button>
        ) : (
          <button
            onClick={stopProcessing}
            style={{
              padding: "8px 20px",
              background: "#ff4d4d22",
              color: "#ff6b6b",
              border: "1px solid #ff4d4d44",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontFamily: "monospace",
            }}
          >
            ■ STOP
          </button>
        )}
      </div>

      {!camRunning && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "14px 18px",
            background: "#0a0a0a",
            border: "1px solid #2a2a2a",
            borderRadius: 8,
            marginBottom: 16,
          }}
        >
          <span style={{ fontSize: 22, color: "#aaa" }}>◎</span>
          <div>
            <div
              style={{ fontSize: 12, color: "#fff", fontFamily: "monospace" }}
            >
              Camera not started
            </div>
            <div style={{ fontSize: 11, color: "#aaa", marginTop: 3 }}>
              Ensure your webcam is connected and click START CAMERA. Make sure
              no other app is using it.
            </div>
          </div>
        </div>
      )}

      {/* tabs — shown always so user can see them before/after starting */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
        {[
          ["live", "LIVE VIEW"],
          ["graphs", "GRAPHS"],
          ["stats", "STATS"],
        ].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setCamTab(id)}
            style={{
              padding: "5px 14px",
              fontSize: 11,
              fontFamily: "monospace",
              background: camTab === id ? "#00e5a022" : "#111",
              color: camTab === id ? "#00e5a0" : "#aaa",
              border: `1px solid ${camTab === id ? "#00e5a044" : "#2a2a2a"}`,
              borderRadius: 5,
              cursor: "pointer",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {camTab === "live" && (
        <>
          <div style={{ display: "flex", gap: 14, marginBottom: 16 }}>
            <VideoFeed
              label="● LIVE INPUT"
              accent="#ff4d4d"
              running={camRunning}
              streamPath="/video_feed/raw"
            />
            <VideoFeed
              label="◆ STABILIZED"
              accent="#00e5a0"
              running={camRunning}
              streamPath="/video_feed/stabilized"
            />
          </div>
          {camRunning && (
            <div style={{ marginBottom: 16 }}>
              <HUD metrics={metrics} />
            </div>
          )}
          <Panel title="Transform Mode">
            <ModeBar mode={mode} onChange={changeMode} />
          </Panel>
        </>
      )}

      {camTab === "graphs" &&
        (camRunning ? (
          <Graphs H={H} />
        ) : (
          <div style={{ color: "#aaa", fontSize: 12, fontFamily: "monospace" }}>
            Start the camera to see live graphs.
          </div>
        ))}

      {camTab === "stats" &&
        (camRunning ? (
          <Panel title="Live Stats">
            <HUD metrics={metrics} />
          </Panel>
        ) : (
          <div style={{ color: "#aaa", fontSize: 12, fontFamily: "monospace" }}>
            Start the camera to see stats.
          </div>
        ))}
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("home");
  const [videos, setVideos] = useState([]);
  const [mode, setMode] = useState("final");
  const [running, setRunning] = useState(false);
  const [source, setSource] = useState(null);
  const [tab, setTab] = useState("live");
  const [metrics, setMetrics] = useState(null);
  const [summaries, setSummaries] = useState({});
  const [error, setError] = useState(null);
  const [apiOk, setApiOk] = useState(false);

  const hist = useRef({
    raw: [],
    stab: [],
    roi_raw: [],
    roi_stab: [],
    fps: [],
    dx: [],
    dy: [],
  });
  const push = (key, val) => {
    hist.current[key] = [...hist.current[key].slice(-199), val];
  };
  const esRef = useRef(null);

  useEffect(() => {
    const check = async () => {
      try {
        await fetch(`${API}/status`);
        setApiOk(true);
      } catch {
        setApiOk(false);
      }
    };
    check();
    const id = setInterval(check, 3000);
    return () => clearInterval(id);
  }, []);

  const fetchVideos = useCallback(async () => {
    try {
      const r = await fetch(`${API}/videos`);
      setVideos(await r.json());
    } catch {
      setError("Cannot reach API — is api.py running?");
    }
  }, []);

  useEffect(() => {
    if (apiOk) fetchVideos();
  }, [apiOk, fetchVideos]);

  const connectSSE = useCallback(() => {
    if (esRef.current) esRef.current.close();
    const es = new EventSource(`${API}/stream`);
    esRef.current = es;
    es.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.type === "metrics") {
          setMetrics(d);
          setRunning(true);
          push("raw", d.raw_disp);
          push("stab", d.stab_disp);
          push("roi_raw", d.roi_raw);
          push("roi_stab", d.roi_stab);
          push("fps", d.fps);
          push("dx", Math.abs(d.dx));
          push("dy", Math.abs(d.dy));
        } else if (d.type === "start") {
          hist.current = {
            raw: [],
            stab: [],
            roi_raw: [],
            roi_stab: [],
            fps: [],
            dx: [],
            dy: [],
          };
          setRunning(true);
          setMetrics(null);
        } else if (d.type === "summary") {
          setSummaries((prev) => ({ ...prev, [d.source]: d }));
          setRunning(false);
        } else if (d.type === "done") {
          setRunning(false);
        } else if (d.type === "error") {
          setError(d.message);
          setRunning(false);
        }
      } catch {}
    };
    es.onerror = () => {};
  }, []);

  useEffect(() => {
    connectSSE();
    return () => esRef.current?.close();
  }, [connectSSE]);

  const startSource = async (src) => {
    setError(null);
    setSource(src);
    hist.current = {
      raw: [],
      stab: [],
      roi_raw: [],
      roi_stab: [],
      fps: [],
      dx: [],
      dy: [],
    };
    try {
      await fetch(`${API}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: src, mode }),
      });
    } catch {
      setError("Cannot reach API");
    }
  };

  const stopProcessing = async () => {
    try {
      await fetch(`${API}/stop`, { method: "POST" });
    } catch {}
    setRunning(false);
  };

  const changeMode = async (m) => {
    setMode(m);
    try {
      await fetch(`${API}/mode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: m }),
      });
    } catch {}
  };

  const H = hist.current;

  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        background: "#060606",
        color: "#ccc",
      }}
    >
      <Sidebar
        page={page}
        setPage={setPage}
        mode={mode}
        changeMode={changeMode}
        apiOk={apiOk}
        running={running}
        source={source}
      />
      <div style={{ flex: 1, overflowY: "auto", maxHeight: "100vh" }}>
        {error && (
          <div
            style={{
              margin: "16px 24px 0",
              padding: "10px 14px",
              background: "#1a0a0a",
              border: "1px solid #ff4d4d44",
              borderRadius: 7,
              fontSize: 12,
              color: "#ff6b6b",
              fontFamily: "monospace",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            ⚠ {error}
            <button
              onClick={() => setError(null)}
              style={{
                background: "none",
                border: "none",
                color: "#ff6b6b",
                cursor: "pointer",
                fontSize: 14,
              }}
            >
              ✕
            </button>
          </div>
        )}
        {page === "home" && (
          <HomePage
            setPage={setPage}
            videos={videos}
            summaries={summaries}
            apiOk={apiOk}
          />
        )}
        {page === "dataset" && (
          <DatasetPage
            videos={videos}
            running={running}
            source={source}
            metrics={metrics}
            summaries={summaries}
            mode={mode}
            tab={tab}
            setTab={setTab}
            H={H}
            startSource={startSource}
            stopProcessing={stopProcessing}
            changeMode={changeMode}
            fetchVideos={fetchVideos}
          />
        )}
        {page === "camera" && (
          <CameraPage
            running={running}
            source={source}
            metrics={metrics}
            mode={mode}
            H={H}
            startSource={startSource}
            stopProcessing={stopProcessing}
            changeMode={changeMode}
          />
        )}
      </div>
    </div>
  );
}
