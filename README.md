# 🛡️ SpamShield — Email Spam Detector

React UI + Python ML backend.

| Layer | Tech |
|---|---|
| UI | React 18 + Vite, glassmorphism dark theme |
| API | Flask |
| Classic ML | **Naive Bayes written from scratch** (Bayes theorem, Laplace smoothing, log-space) |
| NLP | cleaning, URL/number tokens, stop-words, stemming, TF-IDF (1-2 grams) |
| Deep learning | Multi-layer neural network (MLP 128→64) on TF-IDF features |

```
email-spam-detector/
├── backend/
│   ├── data/emails.csv        # demo dataset (generated)
│   ├── models/                # trained models land here
│   ├── nlp_utils.py           # NLP preprocessing
│   ├── naive_bayes.py         # Bayes theorem from scratch
│   ├── generate_data.py       # makes demo dataset
│   ├── train.py               # trains + evaluates all models
│   ├── app.py                 # Flask API
│   └── requirements.txt
└── frontend/                  # React app (Vite)
```

## Run in VS Code (step by step)

Prerequisites: Python 3.10+, Node.js 18+, Git, VS Code.

1. **Open the folder**: VS Code → File → Open Folder → `email-spam-detector`.
2. **Open two terminals** (Terminal → New Terminal, then the `+` button / split).

**Terminal 1 — backend**
```bash
cd backend
python -m venv venv
# Windows:  venv\Scripts\activate        Mac/Linux:  source venv/bin/activate
pip install -r requirements.txt
python generate_data.py      # only for demo data
python train.py              # trains the models (takes seconds)
python app.py                # API on http://127.0.0.1:5000
```

**Terminal 2 — frontend**
```bash
cd frontend
npm install
npm run dev                  # open http://localhost:5173
```

In VS Code, pick the interpreter with Ctrl+Shift+P → "Python: Select Interpreter" → the `venv` one.

## Use a real dataset (important)

The bundled data is synthetic, so it scores ~100%, which is not realistic. For real results, download the
**SMS Spam Collection** (`spam.csv`) from Kaggle, put it in `backend/data/`, then:
```bash
python train.py data/spam.csv
```
Expect about 97–98% accuracy for Naive Bayes. Columns `v1/v2` or `label/text` are both supported.

## Push to GitHub (step by step)

1. Create an empty repo on github.com (no README) — e.g. `email-spam-detector`.
2. In the project root terminal:
```bash
git init
git add .
git commit -m "Initial commit: email spam detector"
git branch -M main
git remote add origin https://github.com/<your-username>/email-spam-detector.git
git push -u origin main
```
3. First time using Git? Set your identity once:
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
4. Later changes: `git add . && git commit -m "message" && git push`.
   (Or use VS Code's Source Control tab: stage → message → Commit → Sync.)

Trained `.joblib` files are git-ignored; anyone cloning just runs `python train.py`.

## How it works

Bayes theorem: `P(spam | words) ∝ P(spam) · ∏ P(wordᵢ | spam)`. Each word's likelihood is estimated from
training counts with Laplace smoothing; log probabilities avoid underflow. The final verdict averages
Naive Bayes and the neural network. The "Spammy words" chips come from each word's log-likelihood ratio.

## Ideas to extend
Swap the MLP for an LSTM/BERT (TensorFlow or PyTorch), add `.eml` upload, Docker, or deploy on Render + Vercel.
