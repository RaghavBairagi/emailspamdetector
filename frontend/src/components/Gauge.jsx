// Semi-circular spam-probability gauge.
export default function Gauge({ score }) {
  const pct = Math.round(score * 100);
  const r = 80;
  const arc = Math.PI * r; // half circumference
  const color = score >= 0.5 ? "var(--spam)" : "var(--ham)";
  return (
    <svg viewBox="0 0 200 120" className="gauge" role="img" aria-label={`Spam probability ${pct}%`}>
      <path d="M20 100 A80 80 0 0 1 180 100" className="gauge-track" />
      <path
        d="M20 100 A80 80 0 0 1 180 100"
        className="gauge-fill"
        style={{ stroke: color, strokeDasharray: `${arc * score} ${arc}` }}
      />
      <text x="100" y="92" textAnchor="middle" className="gauge-num">{pct}%</text>
      <text x="100" y="112" textAnchor="middle" className="gauge-label">spam probability</text>
    </svg>
  );
}
