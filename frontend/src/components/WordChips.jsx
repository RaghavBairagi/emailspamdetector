// Shows which words influenced the Bayes decision.
export default function WordChips({ title, words, kind }) {
  if (!words.length) return null;
  return (
    <div className="chips-block">
      <h4>{title}</h4>
      <div className="chips">
        {words.map((w) => (
          <span key={w.word} className={`chip ${kind}`}>{w.word}</span>
        ))}
      </div>
    </div>
  );
}
