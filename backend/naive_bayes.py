"""Multinomial Naive Bayes written from scratch.

Bayes theorem:   P(spam | words) = P(words | spam) * P(spam) / P(words)
Naive assumption: words are independent given the class, so
    P(words | spam) = prod_i P(word_i | spam)
We work in log-space to avoid numeric underflow and use Laplace (add-alpha) smoothing.
"""
import math
from collections import Counter, defaultdict

from nlp_utils import preprocess


class NaiveBayesSpam:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes = ("ham", "spam")
        self.word_counts = {c: Counter() for c in self.classes}
        self.class_docs = Counter()
        self.vocab: set[str] = set()
        self.total_words = {c: 0 for c in self.classes}

    def fit(self, texts, labels):
        for text, label in zip(texts, labels):
            tokens = preprocess(text)
            self.class_docs[label] += 1
            self.word_counts[label].update(tokens)
            self.total_words[label] += len(tokens)
            self.vocab.update(tokens)
        return self

    def _log_prior(self, c):
        return math.log(self.class_docs[c] / sum(self.class_docs.values()))

    def _log_likelihood(self, word, c):
        num = self.word_counts[c][word] + self.alpha
        den = self.total_words[c] + self.alpha * len(self.vocab)
        return math.log(num / den)

    def predict_proba(self, text):
        tokens = [t for t in preprocess(text) if t in self.vocab]
        scores = {c: self._log_prior(c) + sum(self._log_likelihood(t, c) for t in tokens)
                  for c in self.classes}
        m = max(scores.values())                       # log-sum-exp for normalisation
        exp = {c: math.exp(s - m) for c, s in scores.items()}
        z = sum(exp.values())
        return {c: exp[c] / z for c in self.classes}

    def predict(self, text):
        p = self.predict_proba(text)
        return max(p, key=p.get)

    def explain(self, text, top_k=6):
        """Words that pushed the decision most towards spam / ham (log-likelihood ratio)."""
        seen = defaultdict(int)
        for t in preprocess(text):
            if t in self.vocab:
                seen[t] += 1
        rows = [(w, self._log_likelihood(w, "spam") - self._log_likelihood(w, "ham")) for w in seen]
        rows.sort(key=lambda r: r[1], reverse=True)
        spammy = [{"word": w, "weight": round(s, 2)} for w, s in rows if s > 0][:top_k]
        hammy = [{"word": w, "weight": round(s, 2)} for w, s in reversed(rows) if s < 0][:top_k]
        return {"spam_words": spammy, "ham_words": hammy}
