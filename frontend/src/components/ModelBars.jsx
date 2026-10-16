// Side-by-side comparison of each model's spam probability.
export default function ModelBars({ models }) {
  const rows = [
    ["Naive Bayes", models.naive_bayes.spam],
    ["Neural Network", models.neural_network.spam],
  ];
  return (
    <div className="bars">
      {rows.map(([name, v]) => (
        <div key={name} className="bar-row">
          <span>{name}</span>
          <div className="bar"><i style={{ width: `${v * 100}%` }} /></div>
          <b>{(v * 100).toFixed(1)}%</b>
        </div>
      ))}
    </div>
  );
}
