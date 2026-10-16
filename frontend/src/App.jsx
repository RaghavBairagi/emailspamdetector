import { useState } from "react";
import Gauge from "./components/Gauge.jsx";
import ModelBars from "./components/ModelBars.jsx";
import WordChips from "./components/WordChips.jsx";

const SAMPLES = {
  "🎁 Prize scam": "Congratulations! You have won a $500 gift card. Click http://win-prize.biz/claim to claim your prize now",
  "💼 Work mail": "Hi Priya, please find attached the report for the budget meeting. Let me know your feedback by Friday.",
  "🏦 Phishing": "URGENT: your account has been suspended. Verify your password at www.secure-bank-login.net immediately",
};

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyze() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setResult(data);
    } catch (e) {
      setResult(null);
      setError(e.message === "Failed to fetch" ? "Cannot reach the backend. Is Flask running on port 5000?" : e.message);
    } finally {
      setLoading(false);
    }
  }

  const isSpam = result?.verdict === "spam";

  return (
    <div className="page">
      <div className="blob b1" />
      <div className="blob b2" />
      <main className="shell">
        <header>
          <span className="logo">🛡️</span>
          <h1>SpamShield</h1>
          <p>Paste an email. Naive Bayes and a neural network will tell you whether it's spam.</p>
        </header>

        <section className="card">
          <div className="samples">
            {Object.entries(SAMPLES).map(([k, v]) => (
              <button key={k} className="ghost" onClick={() => setText(v)}>{k}</button>
            ))}
          </div>
          <textarea
            rows={7}
            placeholder="Paste your email text here…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button className="primary" disabled={loading || !text.trim()} onClick={analyze}>
            {loading ? "Analyzing…" : "Analyze email"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        {result && (
          <section className={`card result ${isSpam ? "is-spam" : "is-ham"}`}>
            <div className="verdict">
              <Gauge score={result.spam_score} />
              <div>
                <span className="badge">{isSpam ? "Spam detected" : "Looks legitimate"}</span>
                <p className="hint">
                  {isSpam ? "Be careful — don't click links or share personal details." : "No strong spam signals found."}
                </p>
              </div>
            </div>
            <ModelBars models={result.models} />
            <WordChips title="Spammy words" words={result.explanation.spam_words} kind="spam" />
            <WordChips title="Trustworthy words" words={result.explanation.ham_words} kind="ham" />
          </section>
        )}

        <footer>React · Flask · Naive Bayes · NLP · Neural Network</footer>
      </main>
    </div>
  );
}
