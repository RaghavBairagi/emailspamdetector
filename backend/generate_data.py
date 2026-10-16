"""Creates a synthetic demo dataset (data/emails.csv) so the project runs out of the box.
For real accuracy, replace it with a real corpus (see README: Kaggle 'spam.csv')."""
import csv
import random

random.seed(42)

SPAM = [
    "Congratulations! You have won a ${n} gift card. Click {url} to claim your prize now",
    "URGENT: your account has been suspended. Verify your password at {url} immediately",
    "Get cheap {drug} online, no prescription needed, limited offer, order now",
    "You are selected for a free {item}! Reply YES to claim. Offer expires today",
    "Make ${n} a day working from home. No experience needed. Sign up at {url}",
    "Final notice: claim your lottery winnings of ${n},000. Send your bank details today",
    "Hot singles in your area want to meet you. Click {url} now, free membership",
    "Exclusive deal!!! 90% off {item}. Buy now, act fast, limited stock, {url}",
    "Your loan of ${n},000 is pre-approved. No credit check. Apply now at {url}",
    "FREE entry to win a {item}. Text WIN to {n}{n}{n}{n}. T&Cs apply, congratulations",
    "Dear customer, your PayPal is locked. Confirm your identity and credit card at {url}",
    "Earn bitcoin fast! Double your investment in 24 hours guaranteed. Invest now {url}",
]
HAM = [
    "Hi {name}, are we still meeting for lunch tomorrow at {t}?",
    "Please find attached the report for the {topic} meeting. Let me know your feedback",
    "Reminder: the {topic} review is scheduled on Monday at {t}. Please bring your notes",
    "Thanks for your help yesterday, {name}. I really appreciate it",
    "Can you send me the notes from the {topic} class? I missed it today",
    "Mom called, she wants us to come home this weekend for dinner",
    "The project deadline has moved to Friday. Let's sync on the {topic} tasks tomorrow",
    "Happy birthday {name}! Hope you have a wonderful day with family and friends",
    "I'll be late to the office, traffic is bad. Start the {topic} call without me",
    "Here is the agenda for the {topic} workshop. See you at {t}",
    "Did you push the latest code to the repository? I want to review the {topic} module",
    "Your invoice for the {topic} subscription is attached. Thank you for your business",
]
FILL = dict(
    n=["100", "500", "1000", "250", "5000"],
    url=["http://win-prize.biz/claim", "www.secure-bank-login.net", "http://bit.ly/free-now", "http://cheap-deals.xyz"],
    drug=["pills", "meds", "vitamins"],
    item=["iPhone", "laptop", "vacation", "TV", "watch"],
    name=["Rahul", "Priya", "Amit", "Sara", "John", "Neha"],
    t=["10am", "1pm", "3:30pm", "5pm"],
    topic=["marketing", "python", "budget", "design", "sales", "machine learning"],
)


class SafeDict(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def render(tpl):
    vals = {k: random.choice(v) for k, v in FILL.items()}
    return tpl.format_map(SafeDict(vals))


rows = []
for _ in range(700):
    rows.append(("ham", render(random.choice(HAM))))
for _ in range(500):
    rows.append(("spam", render(random.choice(SPAM))))
random.shuffle(rows)

with open("data/emails.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["label", "text"])
    w.writerows(rows)
print(f"Wrote {len(rows)} rows to data/emails.csv")
