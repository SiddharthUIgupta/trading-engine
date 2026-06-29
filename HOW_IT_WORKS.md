# How This Trading System Works

A simple explanation of the full system, without jargon.

---

## Every morning before market open

The system checks two things: how scared is the market right now (VIX), and is the market going up or down (SPY trend). Based on that, it decides which of its 4 trading strategies to use today. Nobody has to tell it — it figures this out on its own.

---

## When the market opens

For each stock it's watching, it runs through a small team of AI analysts:

- One looks at the news and overall market mood
- One looks at the company's finances
- One looks at the price chart
- A risk officer reviews all three opinions and makes the final call

Before each meeting, the team is reminded of **what worked and what didn't in similar situations before** — like a trader looking at their journal before making a decision. Lessons that kept leading to losses get quietly dropped. Lessons that kept leading to wins get shown first.

The risk officer also gets told **how accurate each analyst has been historically** — so if the chart analyst has been wrong 60% of the time lately, their opinion gets less weight.

---

## How much to buy

It doesn't just always buy the same fixed amount. It calculates how much to risk based on its actual win rate over the last few hundred trades. If it's been winning more, it bets a bit more. If it's been losing, it bets less. This is called Kelly sizing — the same math professional gamblers and hedge funds use.

It also checks if the stock it wants to buy moves almost identically to something it already owns. If it does, it either sizes down or skips the trade entirely — no point having two bets that are basically the same bet.

---

## During the day

Every 15 minutes it checks open positions. If a stock drops past a certain threshold, it sells. If it hits the target, it sells. No emotion, no second-guessing.

---

## After every trade closes

It runs a quick post-mortem — what signals led to this trade, did it win or lose, why? It extracts 1–3 lessons and saves them. Each lesson has a score. Lessons that show up before wins get rewarded. Lessons that show up before losses get penalised. After enough losses, a lesson gets permanently retired — the system stops using it.

---

## The big picture

Over time, the system builds up a track record of what each analyst tends to get right and wrong, in which market conditions, and uses that to make smarter decisions going forward. It doesn't learn in the AI sense of retraining — it learns the same way a human trader would: by keeping a really detailed journal and actually reading it before the next trade.
