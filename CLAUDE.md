# Trading Engine — Claude Instructions

---

## Session Handoff — 2026-07-12 (PROTECTED)

This section is the starting context for new sessions. Read it before anything else.

### System overview

This is a 5-repo integrated AI trading system. The repos are **not independent** — they form a research-to-execution-to-memory loop. Do not treat them as isolated projects.

```
claude-obsidian ──► Vibe-Trading ──► trading-engine ──► Telegram alerts
      ▲                                     │
      └──────── trade post-mortems ◄────────┘
claudian (role TBD — investigate on first Pi session)
```

**Goal:** Vibe-Trading researches and validates alpha factors → trading-engine executes trades using those factors via the VW bandit → every closed trade writes a post-mortem to claude-obsidian → the wiki feeds context back into the Risk Officer's LLM prompt → loop closes.

---

### What each repo does

**trading-engine** (`~/Projects/trading-engine`) — *this repo, the live executor*
- Two-plane architecture: Alpha Plane (scans + LLM consensus + order intent writes) + Protection Plane (exits + brackets + reconciliation)
- IPC via SQLite WAL — Alpha writes `order_intents` table, Protection reads and executes
- 4-agent LLM consensus: sentiment + fundamental + technical analysts + Risk Officer (Claude Sonnet)
- VW bandit (`analyst_layer/vw_bandit.py`): online contextual bandit learning `track × regime × agent_votes → win_probability`. Currently has 0 examples (fixed in Phase 1 — needs warmup run)
- Strategies: thesis (PF 1.49 — only one with real edge), swing, recovery, ORB equity (disabled, PF 0.92), options (disabled, 30 losses)
- Broker: Alpaca paper/live. DB: `state/trading_engine.sqlite3`
- Services on Pi: `trading-engine-alpha` and `trading-engine-protection` (systemd)

**Vibe-Trading** (`~/Projects/Vibe-Trading`) — *research and backtest lab*
- Source: `HKUDS/Vibe-Trading`, install as package: `pip install vibe-trading-ai` (NOT `pip install -e .`)
- **460 pre-built alpha factors** grouped as: alpha101 (Qlib — 101 price/volume), gtja191 (GTJA — 191 factors), qlib158 (quant lib academic), and additional academic set. All point-in-time safe.
- **Alpha bench**: `from src.tools.alpha_bench_tool import run_alpha_bench` — call with `(universe=tickers, start=..., end=...)`, returns list of `{factor_id, ic, n}` rows. IC = Spearman correlation of factor rank vs next-day return.
- **Factor compute pattern**: `Registry().compute(alpha_id, panel)` where `panel = {ticker: OHLCV_DataFrame}`. Returns a DataFrame indexed by date with ticker columns. Find `Registry` class with: `grep -r "class Registry" ~/Projects/Vibe-Trading/src/`
- **Multi-agent swarm**: natural-language research goals run by an investment committee agent
- **MCP server**: `vibe-trading-mcp` (stdio), 54+ tools — can be added to `.mcp.json` for use in Claude Code sessions
- CLI: `vibe-trading` (interactive REPL), `vibe-trading serve --port 8899` (web UI at localhost:8899)
- Config search order: `~/.vibe-trading/.env` → `agent/.env` → `$CWD/.env`. Interactive setup: `vibe-trading init`
- Needs: `ANTHROPIC_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FINNHUB_API_KEY`
- **Integration point with trading-engine**: promoted factor values → `|factors` namespace in `vw_bandit.py`'s `_full_features()`. See Phase 2 for exact implementation.
- **IC threshold for promotion**: Spearman IC > 0.03 at n≥300 → candidate. Below n=300 = "insufficient sample" — no conclusions. See signal lifecycle rules below.

**claude-obsidian** (`~/Projects/claude-obsidian`) — *persistent knowledge wiki with hybrid retrieval*
- Source: `AgriciDaniel/claude-obsidian` — self-organizing Obsidian vault + Claude Code plugin
- Ingests any source document → extracts concepts/entities → builds cross-linked wiki in `wiki/`
- **Vault structure**: `.raw/` (drop source docs here), `wiki/` (auto-generated pages), `.vault-meta/bm25/` (search index). Read `wiki/hot.md` first, then `wiki/index.md`, then domain `_index.md` files.
- **Hybrid retrieval API** (the key integration): `python3 scripts/retrieve.py "your query"` → JSON: `{"candidates": [{"page_path": "wiki/...", "snippet": "..."}]}`. Combines BM25 keyword match with cosine semantic rerank. Top-K=3 is the sweet spot for LLM context injection.
- **Setup (one-time on Pi)**:
  ```bash
  cd ~/Projects/claude-obsidian
  bash bin/setup-vault.sh       # configures Obsidian vault symlinks
  bash bin/setup-retrieve.sh    # builds BM25 index from wiki/ content
  ```
- **Re-index after adding pages**: run `bash bin/setup-retrieve.sh` again (fast, incremental)
- **Skills available inside the claude-obsidian directory**: `/wiki` (scaffold new topic), `/autoresearch [topic]` (3-round web research → files in wiki), `/wiki-retrieve` (ad-hoc search), `/think`, `/canvas`
- **Integration point with trading-engine**:
  - Risk Officer prompt enrichment: `scripts/retrieve.py "TICKER SECTOR REGIME"` → prepend top-3 snippets to Risk Officer system prompt at `analyst_layer/agents/risk_officer_agent.py:44`
  - Post-mortem ingest: after each closed trade, write a markdown file to `vaults/trading/postmortems/TICKER-DATE.md` → BM25 index picks it up → future trades in the same setup get the memory
  - See Phase 3 for exact implementation.

**claudian** (`~/Projects/claudian`) — *unknown — investigate on first Pi session*
- Source: `YishenTu/claudian` — not yet cloned on Mac, role not yet determined
- **First action on Pi**: `cat ~/Projects/claudian/README.md` — read it, understand what it does, then update this CLAUDE.md section via `/preserve` with what you learned and where it fits in the integration plan

**cpr-compress-preserve-resume** (`~/Projects/cpr-compress-preserve-resume`) — *session context preservation*
- Source: `EliaAlberti/cpr-compress-preserve-resume`
- Already installed in this repo as `.claude/commands/` — gives you `/compress`, `/preserve`, `/resume`
- **Use at end of every Pi session**: run `/compress` to save full session log to `CC-Session-Logs/`, then `/preserve` to update this CLAUDE.md with what changed. Next session starts with full context.
- `/resume` loads the most recent session log back into context when starting fresh

---

### Phase 1 — COMPLETE (committed 2026-07-12, commit 8fafd44)

All changes are on `master`. Pull on Pi, then do the Pi-only steps below.

**What was fixed:**
| File | Change |
|------|--------|
| `pyproject.toml` | Added `vowpalwabbit>=9.11.2` (was missing → VW silently disabled on Pi) and `akshare>=1.0.0` (MacroSnapshot broken) |
| `execution_layer/protection_plane.py` | Fixed VW bandit: `learn()` now fires for **every** closed trade — was gated behind `if not agent_signals` early-return, so 0 examples were ever recorded |
| `config/settings.py` | `orb_equity_enabled` default `True→False` (PF 0.92, losing); `options_track_enabled` default `True→False` (30 losing positions); added `telegram_bot_token`, `telegram_chat_id`, `extra_watchlist_tickers` fields |
| `execution_layer/alerting.py` | Added `_send_telegram()` + `_broadcast()` — all 9 alert functions now send to Telegram (primary) + Gmail (optional) |
| `main_alpha.py` | Removed hardcoded `WATCHLIST` (CLAUDE.md violation); wired `GlobalRiskState` into all 4 Alpha Plane `RobustCircuitBreaker` instances (was missing, defence-in-depth gap vs Protection Plane) |
| `.env.example` | Added `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `EXTRA_WATCHLIST_TICKERS` |

**Pi-only steps still required (cannot be done remotely):**
```bash
# ── 0. Clone the other repos (only needed once) ─────────────────────────────
mkdir -p ~/Projects && cd ~/Projects
git clone https://github.com/HKUDS/Vibe-Trading.git
git clone https://github.com/YishenTu/claudian.git
git clone https://github.com/AgriciDaniel/claude-obsidian.git
git clone https://github.com/EliaAlberti/cpr-compress-preserve-resume.git

# ── 1. Pull trading-engine changes ──────────────────────────────────────────
cd ~/Projects/trading-engine && git pull

# ── 2. Add to .env (get bot token from @BotFather on Telegram) ─────────────
#    TELEGRAM_BOT_TOKEN=<token>
#    TELEGRAM_CHAT_ID=<your chat id>
#    EXTRA_WATCHLIST_TICKERS=SHAZ,NNBR,LAES,WULF,MARA   # Sid's manual watchlist

# ── 3. Install new deps ──────────────────────────────────────────────────────
.venv/bin/pip install vowpalwabbit==9.11.2 akshare

# ── 4. Bootstrap VW bandit from trade history (run once before restart) ─────
.venv/bin/python scripts/vw_warmup.py

# ── 5. Restart both services ─────────────────────────────────────────────────
sudo systemctl restart trading-engine-alpha trading-engine-protection
```

### Phase 2 — Next (NOT started)

**Why:** The VW bandit currently learns from only 3 feature namespaces: `track`, `regime`, and `agent_votes`. These are high-level categorical signals. Vibe-Trading has 460 pre-built quantitative alpha factors (momentum, reversal, vol-adjusted, cross-sectional rank) that are point-in-time safe. Adding these as VW feature namespaces gives the bandit richer context → better win-probability estimates → less Kelly-size waste on low-confidence setups. The alpha bench step ensures we only promote factors with proven IC on our actual thesis universe, satisfying the "no strategy without a backtest artifact" invariant.

**Pre-conditions:** Phase 1 complete. VW bandit has ≥20 examples (after warmup script). At least 100 closed thesis trades in the ledger.

**Step 1 — Install and configure Vibe-Trading on Pi**

```bash
# Install as pip package (do NOT pip install -e . — the package is on PyPI)
cd ~/Projects/Vibe-Trading
~/Projects/trading-engine/.venv/bin/pip install vibe-trading-ai

# Interactive setup — writes keys to ~/.vibe-trading/.env
vibe-trading init
# Prompts for: ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, FINNHUB_API_KEY
# Keys must also exist in ~/Projects/trading-engine/.env (already set for the engine)

# Verify install
vibe-trading --help    # should print CLI help
```

**Step 2 — Discover which factors have edge on thesis universe**

Run the alpha bench against the thesis universe (100+ closed trade tickers). This step determines the `PROMOTED_FACTORS` list used in Step 4.

```bash
# Run bench from Vibe-Trading repo (use trading-engine .venv which has the package)
cd ~/Projects/Vibe-Trading
~/Projects/trading-engine/.venv/bin/python - <<'EOF'
from src.tools.alpha_bench_tool import run_alpha_bench
# thesis_universe: list of tickers from your closed trades
# Fetch from DB: SELECT DISTINCT ticker FROM realized_sales
import sqlite3
conn = sqlite3.connect("/home/sid/Projects/trading-engine/state/trading_engine.sqlite3")
tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM realized_sales").fetchall()]
conn.close()

# IC threshold: Spearman IC > 0.03 at n≥300 = candidate for promotion
results = run_alpha_bench(universe=tickers, start="2024-01-01", end="2025-12-31")
# Print factors above threshold
for row in results:
    if row["ic"] > 0.03 and row["n"] >= 300:
        print(row["factor_id"], row["ic"])
EOF
```
Copy the printed factor IDs — these become `PROMOTED_FACTORS` in `settings.py`.

**Step 3 — Add promoted_factors setting to settings.py**

In `config/settings.py`, add alongside the other fields:
```python
promoted_vw_factors: list[str] = Field(
    default_factory=list, alias="PROMOTED_VW_FACTORS"
)

@field_validator("promoted_vw_factors", mode="before")
@classmethod
def _parse_csv_factors(cls, v: object) -> list[str]:
    if isinstance(v, str):
        return [f.strip() for f in v.split(",") if f.strip()]
    return v
```
Then add to `.env`: `PROMOTED_VW_FACTORS=alpha101_001,gtja191_030,...` (from Step 2 output).

**Step 4 — Create factor_provider.py**

Create `analyst_layer/factor_provider.py`:
```python
from __future__ import annotations
import logging, sys
from pathlib import Path
import pandas as pd

_VIBE_PATH = Path.home() / "Projects" / "Vibe-Trading"
_log = logging.getLogger(__name__)

def compute_factor_features(ticker: str, factor_ids: list[str]) -> dict[str, float]:
    """Returns {factor_id: latest_value} for each promoted factor. Empty dict on failure."""
    if not factor_ids or not (_VIBE_PATH / "src").exists():
        return {}
    try:
        sys.path.insert(0, str(_VIBE_PATH))
        from src.alpha.factor_registry import Registry  # confirmed location in Vibe-Trading repo
        reg = Registry()
        # panel = {ticker: OHLCV DataFrame for last 252 trading days}
        # Vibe-Trading's data adapters wrap yfinance — use the same period as backtest
        import yfinance as yf
        raw = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        panel = {ticker: raw}
        out: dict[str, float] = {}
        for fid in factor_ids:
            try:
                result: pd.DataFrame = reg.compute(fid, panel)
                latest = float(result[ticker].dropna().iloc[-1])
                out[fid] = latest
            except Exception as exc:
                _log.debug("factor %s failed for %s: %s", fid, ticker, exc)
        return out
    except Exception as exc:
        _log.warning("factor_provider unavailable: %s", exc)
        return {}
```
If the exact import path `src.alpha.factor_registry.Registry` doesn't exist, run `grep -r "class Registry" ~/Projects/Vibe-Trading/` to find it before writing this file.

**Step 5 — Enrich VW bandit features in vw_bandit.py**

In `analyst_layer/vw_bandit.py`, find the `_full_features()` method. Add a `|factors` namespace after the existing namespaces:

```python
# At top of vw_bandit.py — add import
from analyst_layer.factor_provider import compute_factor_features

# Inside _full_features(self, ctx: dict) — add AFTER existing namespaces:
promoted = getattr(settings, "promoted_vw_factors", [])
if promoted:
    factor_vals = compute_factor_features(ctx.get("ticker", ""), promoted)
    if factor_vals:
        factor_ns = " ".join(f"{k}:{v:.4f}" for k, v in factor_vals.items())
        vw_str += f" |factors {factor_ns}"
```
This adds a `|factors` VW namespace with the promoted factor values. VW learns weights per feature automatically.

**Step 6 — Wire vw_prob into Kelly sizing (alpha_plane.py)**

In `main_alpha.py` or `alpha_plane.py`, find where `vw_bandit.predict_full(ctx)` is called and where Kelly fraction is computed. Currently `predict_full()` is called but the result is unused for sizing. The change:

```python
# After vw_bandit.predict_full(ctx):
vw_prob = vw_bandit.predict_full(ctx)  # 0.0–1.0 win probability

# Find the kelly fraction calculation (grep for "kelly" in alpha_plane.py / main_alpha.py)
# Add a multiplier cap so low-confidence VW predictions reduce size:
# If vw_prob < 0.50 → scale down; vw_prob ≥ 0.55 → full Kelly
VW_CONFIDENCE_THRESHOLD = 0.55
kelly_fraction = kelly_fraction * min(1.0, vw_prob / VW_CONFIDENCE_THRESHOLD)
# Log this so we can audit it: logger.info("VW prob %.2f → kelly scale %.2f", vw_prob, ...)
```
Do NOT implement this step until ≥300 closed trades exist and `scripts/signal_uplift.py` shows VW features have edge. Until then, `predict_full()` runs in shadow mode (logged but not used). This is the "shadow first" invariant.

---

### Phase 3 — Future

**Why:** The Risk Officer makes decisions in a vacuum. It doesn't know that the last 4 times we bought MSFT in a bull regime at resistance, we got stopped out. claude-obsidian is a hybrid-retrieval knowledge system that builds a searchable wiki from trade post-mortems. By wiring it into the Risk Officer system prompt, the LLM gets memory of prior trades in the same setup — reducing repeated mistakes. This closes the loop: live trades → markdown post-mortems → BM25 index → LLM context → better decisions.

**Why claudian matters here (TBD):** YishenTu/claudian's role is not yet known. It may be a Claude agent orchestration framework that could replace or augment the 4-agent consensus. Investigate before Phase 3 begins.

**Pre-conditions:** claude-obsidian cloned on Pi. At least 50 closed trades exist (enough post-mortems to make retrieval useful).

**Step 1 — Set up claude-obsidian trading vault**

```bash
cd ~/Projects/claude-obsidian

# One-time Obsidian config setup (creates .obsidian/ symlinks)
bash bin/setup-vault.sh

# One-time BM25 index build (run again after adding new pages)
bash bin/setup-retrieve.sh

# Create the trading vault directory
mkdir -p vaults/trading/postmortems

# Test retrieval works
python3 scripts/retrieve.py "MSFT thesis bull"
# Should return JSON: {"candidates": [{"page_path": "...", "snippet": "..."}]}
```

**Step 2 — Auto post-mortem writer in protection_plane.py**

In `execution_layer/protection_plane.py`, find `_run_reflection()` — the method called after a position closes to extract lessons. After it runs, add a call to write a structured markdown post-mortem:

Create helper `execution_layer/postmortem_writer.py`:
```python
from __future__ import annotations
import logging, subprocess
from datetime import date
from pathlib import Path

_VAULT = Path.home() / "Projects" / "claude-obsidian" / "vaults" / "trading" / "postmortems"
_RETRIEVE_SCRIPT = Path.home() / "Projects" / "claude-obsidian" / "scripts" / "retrieve.py"
_log = logging.getLogger(__name__)

def write_postmortem(
    ticker: str, strategy: str, regime: str,
    entry: float, exit: float, pnl: float,
    agent_stances: dict[str, str],  # {"sentiment": "BUY", "risk_officer": "PASS"}
    lessons: list[str],
) -> None:
    if not _VAULT.exists():
        return  # vault not set up — skip silently
    date_str = date.today().isoformat()
    outcome = "WIN" if pnl > 0 else "LOSS"
    md = [
        f"# {ticker} {date_str} {outcome}",
        f"\nStrategy: {strategy}  Regime: {regime}",
        f"Entry: {entry:.2f}  Exit: {exit:.2f}  PnL: {pnl:+.2f}",
        "\n## Agent stances",
        *[f"- {agent}: {stance}" for agent, stance in agent_stances.items()],
        "\n## Lessons",
        *[f"- {l}" for l in lessons],
    ]
    path = _VAULT / f"{ticker}-{date_str}.md"
    path.write_text("\n".join(md))
    _log.info("post-mortem written: %s", path)
    # Re-index BM25 so retrieval picks up the new page immediately
    try:
        subprocess.run(["python3", str(_RETRIEVE_SCRIPT.parent.parent / "bin" / "index.py")],
                       capture_output=True, timeout=30)
    except Exception:
        pass  # index update is best-effort; retrieval still works on next full re-index
```

In `protection_plane.py`, after `self._run_reflection(ticker, ...)`, add:
```python
from execution_layer.postmortem_writer import write_postmortem
write_postmortem(
    ticker=ticker, strategy=position.strategy, regime=current_regime,
    entry=position.avg_entry_price, exit=exit_price, pnl=realized_pnl,
    agent_stances=agent_stances_at_entry,  # store these on the position when entering
    lessons=extracted_lessons,
)
```

**Step 3 — Wiki-retrieve in Risk Officer system prompt**

In `analyst_layer/agents/risk_officer_agent.py`, `def system_prompt(self)` starts at line 44. Add a retrieval call before assembling the prompt:

```python
import subprocess, json
from pathlib import Path

def _fetch_trade_memory(ticker: str, sector: str, regime: str) -> str:
    script = Path.home() / "Projects" / "claude-obsidian" / "scripts" / "retrieve.py"
    if not script.exists():
        return ""
    try:
        query = f"{ticker} {sector} {regime} trade setup"
        r = subprocess.run(["python3", str(script), query],
                           capture_output=True, text=True, timeout=5)
        candidates = json.loads(r.stdout).get("candidates", [])[:3]
        if not candidates:
            return ""
        snippets = "\n---\n".join(c["snippet"] for c in candidates)
        return f"\n\n## Prior trade memory (similar setups)\n{snippets}"
    except Exception:
        return ""

# In system_prompt(self):
memory_context = _fetch_trade_memory(self.ticker, self.sector, self.regime)
# Prepend to the existing prompt text — before the "You are the Risk Officer" block
return memory_context + existing_prompt_text
```

**Step 4 — Investigate claudian and update plan**

```bash
cat ~/Projects/claudian/README.md
# Read it. Determine: is it an agent SDK, a Claude wrapper, a UI framework?
# Then run /preserve to update this CLAUDE.md section with what it does and where it fits.
```

If claudian turns out to be an agent orchestration layer, it may be able to replace the manual 4-agent loop in `analyst_layer/` — but do not integrate until you fully understand its threading/async model and confirm it doesn't violate the "Protection never imports LLM clients" invariant.

### Key architecture facts (do not re-derive without reading the code)

- **Only thesis track has a proven backtest edge** (PF 1.49, 58% WR). ORB equity is disabled (PF 0.92). Options disabled (30 losing positions). Thesis + swing + recovery are the active tracks.
- **VW bandit**: `analyst_layer/vw_bandit.py`. `_MIN_EXAMPLES = 20` — predictions suppressed below this count. `warm_start()` bootstraps from DB on first run. After Phase 1 fix + warmup script, should finally accumulate examples.
- **Lesson store**: frozen (`FREEZE_LESSON_INJECTION=True` in settings) until 100 closed trades on fixed config. Do not unfreeze.
- **GlobalRiskState**: DB-level halt for weekly/trailing drawdown. Now wired in both planes (as of Phase 1). The weekly halt resets on Monday; trailing requires manual `reset()`.
- **Two-plane IPC**: Alpha writes `order_intents` table → Protection reads and executes. Never bypass with direct broker calls from Alpha. Never import scanners/LLM into Protection.
- **CLAUDE.md invariant**: no hardcoded ticker lists — universe from OpenBB screens. Extra tickers go in `EXTRA_WATCHLIST_TICKERS` env var.

---

## The most important rule

**If the user says something, do exactly that. Never assume you know better. If anything is unclear, ask a question before writing a single line of code.**

Examples of when to ask instead of assume:
- User says "integrate X" → ask: where should it plug in? what should it replace or supplement?
- User says "fix the universe" → ask: what should the new universe be? screens only, or something else?
- User says "look into X" → ask: do you want me to install it, just research it, or build something with it?
- User says "make it better" → ask: better in what specific way?

Never interpret a vague instruction as permission to redesign, refactor, or add features the user didn't ask for.

---

## Confirm before acting

- **Never implement from a summary, overview, or "take a look at" prompt.** Those are research requests, not build requests.
- Always wait for an explicit go-ahead ("yes", "do it", "build it", "start now") before writing code.
- If the user approves approach A, do not silently implement approach B because you think it's better.

---

## Risk invariants — violating any of these is a critical bug, not a style choice

Each of these was learned from a real production defect. Any diff that weakens one must be called out explicitly and confirmed before writing it.

1. **Breakers gate ENTRIES only. Exits are unconditional.** No halted/tripped/profit-locked state may ever block, skip, or delay a stop-loss or exit check.
2. **Risk guards are monotonic.** No signal — LLM sentiment, news score, anything — may ever LOOSEN a guard (lower effective VIX, raise a cap, re-arm a track). Tighten or no-op only.
3. **Profit lock = no new entries. Never liquidation.** The thesis edge is winners running for weeks; any rule that force-flattens the book is a different, untested strategy.
4. **Never submit a software SELL on a bracketed position without cancelling the resting stop leg first** (`_release_bracket_stop`). Alpaca reserves the shares against the open stop.
5. **A live rule must exist in the backtest, and a backtested rule must exist live.** If a diff adds an execution-time rule (lock, filter, exit) with no backtest counterpart, say so.
6. **No strategy may be armed in config without a linked, passing point-in-time backtest artifact.** Current-constituent universes are banned for anything that buys drawdowns.
7. **No `except Exception` around decision paths.** Adapters return Ok/Empty/Failed; Empty and Failed are different states and Failed alerts. Silence is an alarm, not a default.
8. **Two-process rules:** all Alpha↔Protection IPC goes through DB tables; any state written on an event (trip, lock) must have a write-back/sync path that resets it, or it is a permanent silent halt; SQLite stays in WAL with a busy timeout; Protection never imports scanners, scrapers, or LLM clients.

---

## Financial methodology — when touching strategy, sizing, or backtest code

- Point-in-time membership always; report skipped-ticker counts so residual survivorship is visible.
- No parameter tuning, lesson injection, or Kelly changes on n < 100 closed trades on a frozen config. Log, don't act.
- Slippage assumptions must be justified against the actual live universe (small caps ≠ S&P slippage).
- The candidate ledger is the measurement instrument: every screened candidate gets a row whether traded or not. Never bypass it.
- LLM layers are advisory until the ledger shows uplift of approved vs rejected candidates. "Shadow mode" is the default for anything new.

### Signal lifecycle

A signal (any `analyst_layer/shadow_signals.py` provider) is always in exactly one of three states — no permanent shadow residents:

1. **Shadow** — logged to `signal_values` for every candidate, influences nothing. Default state for anything new.
2. **Promoted** — used in the risk gate. Requires an explicit, separate, approved task — never bundled into the same change that adds the signal.
3. **Deleted** — removed once `scripts/signal_uplift.py` shows no edge at n>=300.

`scripts/signal_uplift.py`'s verdict at n>=300 is authoritative: `PROMOTE-CANDIDATE` or `DELETE-CANDIDATE`. Below n=300, "INSUFFICIENT SAMPLE" — no conclusions, no promotion, no deletion.

Current-snapshot-only signals (no free point-in-time history — e.g. `short_interest`, which only has a ~20-day-stale "current" settlement snapshot, not a queryable historical series like price-based signals) report `median_staleness_days` alongside their verdict. The n>=300 gate is necessary but not sufficient for these — any `PROMOTE-CANDIDATE` verdict with non-zero staleness needs manual review before it's trusted, since the measurement may have leaked future information into the correlation.

---

## Evidence protocol — how to make claims

- Tag claims: **Certain** (read the code / ran it), **[Likely]** (strong inference), **[Guessing]** (filling gaps). If a reply is mostly guessing, say so first.
- **Master rule: a [Guessing] tag on any external or verifiable fact is illegal — it converts into a tool call.** Guessing is legal only for the genuinely unknowable (future returns, behavior that verification failed to settle).
- Never claim behavior of code you haven't opened this session. Cite file and line for bug claims.

### Verification gates — you may not say it until you've done it

| Before asserting…                                   | Required action this session                          |
|-----------------------------------------------------|-------------------------------------------------------|
| What THIS repo does (any behavior claim)            | Open the file. Cite file:line. Never memory, never docs — ARCHITECTURE.md has lied before. |
| Any Alpaca/broker semantic (order classes, cancels, replace, options levels, rate limits) | Fetch docs.alpaca.markets and/or run it against the paper API. Memory of broker APIs is stale by construction. |
| A library works here                                | Install + import on this machine (aarch64). "It should work" is banned — ARM wheels have burned us. |
| Any market fact (price, halt, split, earnings date, corporate action) | Live fetch via the system's own data adapters first, ad-hoc search second. Training data has no date on it. |
| Any number about the strategy (win rate, PF, exposure, P&L) | Query the DB / candidate ledger / run the backtest. Never estimate a number that is sitting in SQLite. |
| Why production did X                                | `journalctl -u trading-engine-*`, the events table, and the DB — logs before code. Code says what CAN happen; logs say what DID. |
| A fix works                                         | Run the test, revert the fix, confirm red, restore, confirm green. |

### Escalation ladder — cheapest sufficient tool, stop at the first that answers

1. grep / read the repo — internal behavior questions end here
2. run code locally (REPL, pytest, scripts/) — running beats reasoning about running
3. query state (sqlite3, journalctl, heartbeat.json) — production questions end here
4. paper Alpaca API call — broker semantics end here; NEVER verify against live
5. web search — external, current, versioned things
6. web fetch primary source — docs > snippets > blogs
7. ask Sid — only for intent/preference; never for anything tools 1–6 can answer

Escalate only when the current rung can't answer. Three failed attempts on one rung = stop and report what you tried; don't thrash.

### Financial opinions require a data pull first

Any financial judgment — sizing, strategy change, "should we exit X", "is this track working" — must be preceded by pulling: current regime (DB), open positions (broker), relevant candidate-ledger stats, and realized P&L for the bucket. An opinion without those four is speculation and must be labeled as such in its first line. Sample-size gate applies to advice too: no recommendation from n < 100 closed trades on frozen config — say "insufficient sample" instead of extrapolating.

When a bucket has a `strategy_version` split (currently: options, via `CURRENT_OPTIONS_STRATEGY_VERSION` in `state_store.py`), pull P&L split by version. Only the current version's stats are decision-relevant. Older versions may be shown for context but must be labeled "prior strategy — not evidence about current agents" and never cited for or against a new trade.

---

## Proving work is actually done

- Show actual running output — never "I've integrated it". `python -c "import X; print(X.__version__)"`, a real call with real output, and `python -m pytest tests/ -q` passing.
- **A regression test must fail when the fix is reverted.** Revert, run, confirm red, restore, confirm green. A test that passes either way is decoration.
- **Every new seam gets a test the day it's born** — new IPC table, new broker call pattern, new process boundary. "All N tests pass" means nothing about code the N tests never touch.
- If a change affects Protection-plane behavior, state whether the paper fire drill (bracketed entry → forced exit → `kill -9` alpha → stop survives at broker) needs re-running.

---

## Architecture constraints

Two-plane, three-layer. Imports flow one direction:

```
data_layer → analyst_layer → execution_layer
Alpha Plane (alpha_plane.py):           scans, LLM, risk gate → writes order_intents
Protection Plane (protection_plane.py): consumes intents, brackets, exits, reconcile
IPC: SQLite (WAL) — order_intents, breaker_state, events
```

- `data_layer` never imports from `analyst_layer` or `execution_layer`
- `analyst_layer` never imports from `execution_layer`
- Protection Plane stays boring: no scrapers, no scanners, no LLM calls except the exit-escalation agent.

Circuit breaker limits (never change without explicit instruction):
- Intraday (momentum/ORB equity/news): 35% daily drawdown cap
- Options (ORB options/vol): 35% daily drawdown cap
- Thesis/recovery: 10% daily drawdown cap
- Swing: 20% daily drawdown cap

---

## What not to do

- Do not add features, refactors, or abstractions beyond what was asked
- Do not add comments that explain what code does — only add a comment if the WHY is non-obvious
- Do not create markdown files or documentation unless explicitly asked
- Do not push to GitHub unless explicitly told to
- Do not restart the trading engine services unless explicitly told to
- Do not add error handling for scenarios that can't happen
- Do not add backwards-compatibility shims for removed code
- Do not hardcode ticker lists anywhere — universe comes from live discovery screens (all 5 OpenBB screens, market-wide); empty screen → warn and skip, never fall back to a list

---

## Stack / environment

- Raspberry Pi 5, aarch64 — some packages (e.g. pyqlib) do not install on ARM, check before promising
- Python 3.13, virtualenv at `.venv/`
- Broker: Alpaca (paper + live), Level 3 options enabled
- DB: `state/trading_engine.sqlite3` (WAL — `-wal`/`-shm` sidecars are normal; checkpoint before file-copy backups)
- Logs: `logs/trading_engine.log`
- Services: `trading-engine-alpha`, `trading-engine-protection` (systemd)
- Key libraries: vowpalwabbit 9.11.2, backtrader 1.9.78, akshare 1.18.64, openbb, yfinance, anthropic

---

## Before marking any task done

1. Run the relevant code and show the output
2. Run `python -m pytest tests/ -q` — must still pass (407 as of 2026-07-03; update this number when it changes)
3. If it touches imports, run `python -c "from module import thing"` to confirm no import errors
4. If it touches exit paths, breakers, brackets, or IPC: confirm `tests/test_exit_priority.py` and `tests/test_protection_plane.py` still pass and state which invariant (1–8 above) the change interacts with
