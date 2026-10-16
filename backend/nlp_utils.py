"""NLP preprocessing: cleaning, tokenizing, stop-word removal, light stemming."""
import re

STOPWORDS = set("""a an the and or but if then else of to in on at by for with about as into from up down out
over under again further once here there when where why how all any both each few more most other some such
no nor not only own same so than too very can will just don should now i me my myself we our you your he him his
she her it its they them their what which who whom this that these those am is are was were be been being have
has had having do does did doing would could""".split())

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
NUM_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[a-z]+")


def stem(word: str) -> str:
    """Tiny suffix-stripping stemmer (keeps the project free of heavy NLP downloads)."""
    for suf in ("ingly", "edly", "ing", "ed", "ly", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def preprocess(text: str) -> list[str]:
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    tokens = TOKEN_RE.findall(text)
    return [stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


def preprocess_to_string(text: str) -> str:
    return " ".join(preprocess(text))
