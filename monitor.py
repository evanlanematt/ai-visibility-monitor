#!/usr/bin/env python3
"""
AI Visibility Monitor
=====================

Checks, on a schedule, whether AI assistants (ChatGPT, Claude, Perplexity)
recommend each client business, saves every answer to SQLite, and builds an
HTML report with the trend, the fixes that were shipped, and what changed.

Commands
--------
  python monitor.py run                 Check every client with every enabled provider
  python monitor.py run --client <id>   Check one client
  python monitor.py report              Write report.html from the saved history
  python monitor.py change <id> "what changed" [--date YYYY-MM-DD] [--targets "..."]
  python monitor.py status              Print the latest numbers for each client

No third-party packages: Python 3.11+ standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import random
import re
import sqlite3
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULT_QUESTIONS = [
    ("What's the best {category} in {city}?", False),
    ("Can you recommend a good {category} near {city}?", False),
    ("Who is the top-rated {category} in {city}?", False),
    ("I need an affordable {category} in {city}. Who should I call?", False),
    ("Which {category} in {city} has the best reviews?", False),
    ("Compare the top {category} options in {city}.", False),
    ("What {category} in {city} would you suggest for a small business?", False),
    ("Is {name} in {city} a good {category}?", True),
]


@dataclass
class Client:
    id: str
    name: str
    category: str
    city: str
    website: str = ""
    aliases: list[str] = field(default_factory=list)
    extra_questions: list[str] = field(default_factory=list)

    def questions(self) -> list[tuple[str, bool]]:
        qs = [(q.format(name=self.name, category=self.category, city=self.city), b) for q, b in DEFAULT_QUESTIONS]
        qs += [(q, False) for q in self.extra_questions]
        return qs


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Config not found: {path}\nCopy config.example.toml to config.toml and fill it in.")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("settings", {})
    cfg.setdefault("providers", {})
    cfg["clients"] = [Client(**c) for c in cfg.get("clients", [])]
    if not cfg["clients"]:
        sys.exit("No [[clients]] in the config.")
    ids = [c.id for c in cfg["clients"]]
    if len(ids) != len(set(ids)):
        sys.exit("Each client needs a unique id.")
    return cfg


# --------------------------------------------------------------------------
# HTTP helper (stdlib only)
# --------------------------------------------------------------------------

def post_json(url: str, body: dict, headers: dict, timeout: int = 120, retries: int = 2) -> dict:
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            last = RuntimeError(f"HTTP {e.code} from {urlparse(url).netloc}: {detail}")
            if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(5 * (attempt + 1))
                continue
            raise last
        except urllib.error.URLError as e:
            last = RuntimeError(f"Network error reaching {urlparse(url).netloc}: {e.reason}")
            if attempt < retries:
                time.sleep(3)
                continue
            raise last
    raise last  # pragma: no cover


def domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""


ASK_PROMPT = ("A customer asks you the question below. Answer the way you normally would, "
              "naming specific businesses where you can. Keep it under 200 words.\n\nQuestion: {q}")


@dataclass
class Answer:
    text: str
    sources: list[str]  # domains cited


# --------------------------------------------------------------------------
# Providers. Each takes a question and returns the answer text + cited domains.
# API shapes change: if a provider errors, check its docs and adjust parse().
# --------------------------------------------------------------------------

class Provider:
    name = "base"

    def __init__(self, conf: dict):
        self.conf = conf

    def ask(self, question: str) -> Answer:
        raise NotImplementedError


class OpenAIProvider(Provider):
    """ChatGPT-style answers via the OpenAI Responses API with the web search tool."""
    name = "chatgpt"

    def ask(self, question: str) -> Answer:
        body = {"model": self.conf.get("model", "gpt-5"),
                "tools": [{"type": "web_search"}],
                "input": ASK_PROMPT.format(q=question)}
        data = post_json("https://api.openai.com/v1/responses", body,
                         {"Authorization": f"Bearer {self.conf['api_key']}"})
        texts, srcs = [], []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
                    for a in c.get("annotations", []) or []:
                        if a.get("url"):
                            srcs.append(domain(a["url"]))
        return Answer("\n".join(texts).strip(), dedupe(srcs))


class AnthropicProvider(Provider):
    """Claude answers via the Messages API with the web search tool."""
    name = "claude"

    def ask(self, question: str) -> Answer:
        body = {"model": self.conf.get("model", "claude-sonnet-5-5"),
                "max_tokens": 1024,
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                "messages": [{"role": "user", "content": ASK_PROMPT.format(q=question)}]}
        data = post_json("https://api.anthropic.com/v1/messages", body,
                         {"x-api-key": self.conf["api_key"], "anthropic-version": "2023-06-01"})
        texts, srcs = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
                for cit in block.get("citations", []) or []:
                    if cit.get("url"):
                        srcs.append(domain(cit["url"]))
        return Answer("".join(texts).strip(), dedupe(srcs))


class PerplexityProvider(Provider):
    """Perplexity answers via its chat completions API (search is built in)."""
    name = "perplexity"

    def ask(self, question: str) -> Answer:
        body = {"model": self.conf.get("model", "sonar"),
                "messages": [{"role": "user", "content": ASK_PROMPT.format(q=question)}]}
        data = post_json("https://api.perplexity.ai/chat/completions", body,
                         {"Authorization": f"Bearer {self.conf['api_key']}"})
        text = data["choices"][0]["message"]["content"]
        urls = data.get("citations") or [r.get("url", "") for r in data.get("search_results", []) or []]
        return Answer(text.strip(), dedupe(domain(u) for u in urls if u))


class MockProvider(Provider):
    """Offline stand-in for testing the pipeline without API keys or network."""
    name = "mock"
    POOL = ["Capital City Signworks", "Arrow Sign Co.", "Olentangy Graphics", "Buckeye Banner & Sign"]

    def __init__(self, conf: dict, client_name: str = "", boost: float = 0.0):
        super().__init__(conf)
        self.client_name, self.boost = client_name, boost

    def ask(self, question: str) -> Answer:
        rng = random.Random(question + str(time.time_ns()))
        names = rng.sample(self.POOL, 3)
        if self.client_name and rng.random() < 0.25 + self.boost:
            names.insert(rng.randint(0, 2), self.client_name)
        if self.client_name and self.client_name in question:
            return Answer(f"{self.client_name} is a local option with decent reviews.", ["yelp.com"])
        text = "Here are some options:\n" + "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
        return Answer(text, rng.sample(["yelp.com", "expertise.com", "bbb.org", "reddit.com", "threebestrated.com"], 3))


def dedupe(items) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


PROVIDER_CLASSES = {"chatgpt": OpenAIProvider, "claude": AnthropicProvider, "perplexity": PerplexityProvider}
ENV_KEYS = {"chatgpt": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY", "perplexity": "PERPLEXITY_API_KEY"}


def build_providers(cfg: dict, mock: bool) -> list[Provider]:
    if mock:
        return [MockProvider({})]
    out = []
    for name, cls in PROVIDER_CLASSES.items():
        conf = dict(cfg["providers"].get(name, {}))
        if not conf.get("enabled", False):
            continue
        conf["api_key"] = conf.get("api_key") or os.environ.get(ENV_KEYS[name], "")
        if not conf["api_key"]:
            print(f"  ! {name}: enabled but no API key (set {ENV_KEYS[name]}); skipping")
            continue
        out.append(cls(conf))
    if not out:
        sys.exit("No providers ready. Enable at least one in config.toml and set its API key, or use --mock.")
    return out


# --------------------------------------------------------------------------
# Extraction: which businesses did the answer name, in what order?
# --------------------------------------------------------------------------

def norm(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(the|inc|llc|co|company|corp|ltd)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def same_biz(a: str, b: str) -> bool:
    x, y = norm(a), norm(b)
    return bool(x and y and (x == y or x in y or y in x))


EXTRACT_PROMPT = """You are reading answers an AI assistant gave to customer questions.
Target business: "{name}" (also known as: {aliases}).
For each answer, list the specific businesses it recommends or names, in order of first appearance.
Exclude platforms and directories (Google, Yelp, Angi, Thumbtack, BBB, Nextdoor).
Say whether the target is named (allow small spelling differences), its 1-based position in your list,
and for answers that name the target, the tone toward it: positive, neutral, negative,
or unknown if the answer says it doesn't know the business.
Return JSON only: {{"results":[{{"i":0,"businesses":["A","B"],"target_mentioned":false,"target_position":null,"sentiment":null}}]}}

Answers:
{answers}"""


def extract(client: Client, items: list[dict], cfg: dict, mock: bool) -> None:
    """Fills businesses / mentioned / position / sentiment on each item in place."""
    parsed = None
    if not mock:
        parsed = llm_extract(client, items, cfg)
    for i, it in enumerate(items):
        row = (parsed or {}).get(i)
        names = [client.name] + client.aliases
        if row:
            it["businesses"] = [str(b) for b in row.get("businesses", [])][:12]
            idx = next((k for k, b in enumerate(it["businesses"]) if any(same_biz(b, n) for n in names)), -1)
            it["mentioned"] = bool(row.get("target_mentioned")) or idx >= 0
            pos = row.get("target_position")
            it["position"] = pos if isinstance(pos, int) else (idx + 1 if idx >= 0 else None)
            it["sentiment"] = row.get("sentiment") or ("unknown" if it["branded"] else None)
        else:
            heuristic_extract(client, it)


def heuristic_extract(client: Client, it: dict) -> None:
    """Fallback with no LLM: numbered/bulleted list items and bold names, plus a text match for the client."""
    text = it["answer"]
    cands = re.findall(r"^\s*(?:\d+[.)]|[-*•])\s*\**([A-Z][^\n:*–—-]{2,60})", text, flags=re.M)
    cands += re.findall(r"\*\*([^*\n]{3,60})\*\*", text)
    it["businesses"] = dedupe(c.strip(" .") for c in cands)[:12]
    names = [client.name] + client.aliases
    idx = next((k for k, b in enumerate(it["businesses"]) if any(same_biz(b, n) for n in names)), -1)
    in_text = any(norm(n) and norm(n) in norm(text) for n in names)
    it["mentioned"] = idx >= 0 or in_text
    it["position"] = idx + 1 if idx >= 0 else None
    if it["branded"]:
        t = text.lower()
        it["sentiment"] = "unknown" if ("don't have" in t or "not familiar" in t or "no information" in t) else "neutral"
    else:
        it["sentiment"] = None


def llm_extract(client: Client, items: list[dict], cfg: dict) -> dict[int, dict] | None:
    prompt = EXTRACT_PROMPT.format(name=client.name, aliases=", ".join(client.aliases) or "none",
                                   answers=json.dumps([{"i": i, "question": it["question"], "answer": it["answer"]}
                                                       for i, it in enumerate(items)], indent=1))
    p = cfg["providers"]
    try:
        key = p.get("claude", {}).get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            data = post_json("https://api.anthropic.com/v1/messages",
                             {"model": cfg["settings"].get("extract_model_claude", "claude-haiku-5-5"), "max_tokens": 4096,
                              "messages": [{"role": "user", "content": prompt}]},
                             {"x-api-key": key, "anthropic-version": "2023-06-01"})
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        else:
            key = p.get("chatgpt", {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not key:
                return None
            data = post_json("https://api.openai.com/v1/responses",
                             {"model": cfg["settings"].get("extract_model_openai", "gpt-5-mini"), "input": prompt},
                             {"Authorization": f"Bearer {key}"})
            text = "".join(c.get("text", "") for o in data.get("output", []) if o.get("type") == "message"
                           for c in o.get("content", []) if c.get("type") == "output_text")
        m = re.search(r"\{.*\}", text, flags=re.S)
        rows = json.loads(m.group(0))["results"] if m else []
        return {int(r["i"]): r for r in rows if "i" in r}
    except Exception as e:  # extraction is best-effort; fall back to heuristics
        print(f"  ! extraction via LLM failed ({e}); using text matching")
        return None


# --------------------------------------------------------------------------
# Storage (SQLite)
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY, client_id TEXT, provider TEXT, run_at TEXT,
  hits INTEGER, total INTEGER, rate REAL, avg_position REAL, brand_sentiment TEXT
);
CREATE TABLE IF NOT EXISTS answers (
  id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), q_index INTEGER, repeat INTEGER,
  question TEXT, branded INTEGER, answer TEXT, businesses TEXT, mentioned INTEGER,
  position INTEGER, sentiment TEXT, sources TEXT
);
CREATE TABLE IF NOT EXISTS changes (
  id INTEGER PRIMARY KEY, client_id TEXT, date TEXT, what TEXT, targets TEXT
);
CREATE INDEX IF NOT EXISTS runs_client ON runs(client_id, run_at);
"""


def db_connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


# --------------------------------------------------------------------------
# One check = every question x repeats for one client on one provider
# --------------------------------------------------------------------------

def summarize(items: list[dict], client: Client) -> dict:
    """Mention rate is the share of (question, repeat) answers that name the client,
    over unbranded questions only. Share of voice counts answers naming each business."""
    unb = [it for it in items if not it["branded"]]
    hits = [it for it in unb if it["mentioned"]]
    pos = [it["position"] for it in hits if isinstance(it["position"], int)]
    counts: dict[str, int] = {}
    for it in unb:
        seen = set()
        for b in it["businesses"]:
            key = next((k for k in counts if same_biz(k, b)), b)
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
    brand = [it.get("sentiment") for it in items if it["branded"] and it.get("sentiment")]
    return {
        "hits": len(hits), "total": len(unb),
        "rate": round(100 * len(hits) / len(unb), 1) if unb else 0.0,
        "avg_position": round(sum(pos) / len(pos), 2) if pos else None,
        "brand_sentiment": max(set(brand), key=brand.count) if brand else None,
        "sov": sorted(counts.items(), key=lambda kv: -kv[1])[:8],
    }


def check_client(client: Client, provider: Provider, cfg: dict, con: sqlite3.Connection, mock: bool) -> dict:
    repeats = int(cfg["settings"].get("repeats", 3))
    pause = float(cfg["settings"].get("pause_seconds", 1.0))
    if isinstance(provider, MockProvider):
        provider.client_name = client.name
    items = []
    for qi, (q, branded) in enumerate(client.questions()):
        for rep in range(1 if branded else repeats):
            try:
                ans = provider.ask(q)
            except Exception as e:
                print(f"    ! {provider.name} failed on Q{qi + 1}: {e}")
                ans = Answer("", [])
            items.append({"q_index": qi, "repeat": rep, "question": q, "branded": branded,
                          "answer": ans.text, "sources": ans.sources})
            if not mock:
                time.sleep(pause)
    extract(client, [it for it in items if it["answer"]], cfg, mock)
    for it in items:  # failed calls count as not named
        it.setdefault("businesses", []); it.setdefault("mentioned", False)
        it.setdefault("position", None); it.setdefault("sentiment", None)
    s = summarize(items, client)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cur = con.execute("INSERT INTO runs (client_id, provider, run_at, hits, total, rate, avg_position, brand_sentiment) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (client.id, provider.name, now, s["hits"], s["total"], s["rate"], s["avg_position"], s["brand_sentiment"]))
    run_id = cur.lastrowid
    con.executemany("INSERT INTO answers (run_id, q_index, repeat, question, branded, answer, businesses, mentioned, "
                    "position, sentiment, sources) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [(run_id, it["q_index"], it["repeat"], it["question"], int(it["branded"]), it["answer"],
                      json.dumps(it["businesses"]), int(it["mentioned"]), it["position"], it["sentiment"],
                      json.dumps(it["sources"])) for it in items])
    con.commit()
    return {"run_id": run_id, **s}


# --------------------------------------------------------------------------
# Comparisons: what moved since the previous check?
# --------------------------------------------------------------------------

def per_question(con, run_id: int) -> dict[int, dict]:
    rows = con.execute("SELECT q_index, question, branded, mentioned, position FROM answers WHERE run_id=?", (run_id,)).fetchall()
    out: dict[int, dict] = {}
    for r in rows:
        d = out.setdefault(r["q_index"], {"question": r["question"], "branded": bool(r["branded"]), "n": 0, "hits": 0, "pos": []})
        d["n"] += 1
        d["hits"] += r["mentioned"]
        if r["position"]:
            d["pos"].append(r["position"])
    return out


def diff_runs(con, client_id: str, provider: str) -> dict | None:
    runs = con.execute("SELECT * FROM runs WHERE client_id=? AND provider=? ORDER BY run_at DESC LIMIT 2",
                       (client_id, provider)).fetchall()
    if not runs:
        return None
    latest = runs[0]
    prev = runs[1] if len(runs) > 1 else None
    out = {"latest": dict(latest), "prev": dict(prev) if prev else None, "gained": [], "lost": []}
    if prev:
        a, b = per_question(con, latest["id"]), per_question(con, prev["id"])
        for qi, d in a.items():
            if d["branded"] or qi not in b:
                continue
            now_rate, was_rate = d["hits"] / d["n"], b[qi]["hits"] / b[qi]["n"]
            if now_rate - was_rate >= 0.5:
                out["gained"].append(d["question"])
            elif was_rate - now_rate >= 0.5:
                out["lost"].append(d["question"])
    return out


# --------------------------------------------------------------------------
# Notifications (optional): post a summary to a Slack/Discord-style webhook
# --------------------------------------------------------------------------

def notify(cfg: dict, lines: list[str]) -> None:
    url = cfg["settings"].get("webhook_url") or os.environ.get("MONITOR_WEBHOOK_URL")
    if not url or not lines:
        return
    try:
        post_json(url, {"text": "\n".join(lines), "content": "\n".join(lines)}, {}, timeout=20, retries=0)
    except Exception as e:
        print(f"  ! webhook failed: {e}")


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------

def trend_svg(points: dict[str, list[tuple[str, float]]], changes: list[sqlite3.Row]) -> str:
    W, H, L, R, T, B = 760, 240, 46, 90, 18, 34
    all_t = [dt.datetime.fromisoformat(t).timestamp() for s in points.values() for t, _ in s]
    all_t += [dt.datetime.fromisoformat(c["date"] + "T12:00:00+00:00").timestamp() for c in changes]
    if not all_t:
        return "<p class='muted'>No checks yet.</p>"
    t0, t1 = min(all_t), max(all_t)
    if t1 - t0 < 7 * 86400:
        t0, t1 = t0 - 4 * 86400, t1 + 4 * 86400
    x = lambda t: L + (W - L - R) * (t - t0) / (t1 - t0)
    y = lambda v: T + (H - T - B) * (1 - v / 100)
    g = []
    for v in (0, 25, 50, 75, 100):
        g.append(f"<line x1='{L}' x2='{W-R}' y1='{y(v):.1f}' y2='{y(v):.1f}' class='grid'/>"
                 f"<text x='{L-8}' y='{y(v)+4:.1f}' text-anchor='end' class='axis'>{v}%</text>")
    for i, c in enumerate(changes, 1):
        cx = x(dt.datetime.fromisoformat(c["date"] + "T12:00:00+00:00").timestamp())
        g.append(f"<line x1='{cx:.1f}' x2='{cx:.1f}' y1='{T}' y2='{H-B}' class='mark'><title>{html.escape(c['date'] + ': ' + c['what'])}</title></line>"
                 f"<circle cx='{cx:.1f}' cy='{T+8}' r='9' class='markdot'/><text x='{cx:.1f}' y='{T+12}' text-anchor='middle' class='marknum'>{i}</text>")
    colors = {"chatgpt": "#10a37f", "claude": "#c2643a", "perplexity": "#2a55c4", "mock": "#2a55c4"}
    for prov, s in points.items():
        col = colors.get(prov, "#555")
        pts = [(x(dt.datetime.fromisoformat(t).timestamp()), y(v), v) for t, v in s]
        d = " ".join(f"{'L' if i else 'M'}{px:.1f} {py:.1f}" for i, (px, py, _) in enumerate(pts))
        g.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='2.5'/>")
        for i, (px, py, v) in enumerate(pts):
            last = i == len(pts) - 1
            g.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{5 if last else 3}' fill='{col if last else '#fff'}' stroke='{col}' stroke-width='2'><title>{v:.0f}%</title></circle>")
        px, py, v = pts[-1]
        g.append(f"<text x='{px+9:.1f}' y='{py+4:.1f}' class='endlbl' fill='{col}'>{prov} {v:.0f}%</text>")
    return f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Mention rate over time'>{''.join(g)}</svg>"


REPORT_CSS = """
body{font-family:-apple-system,'Segoe UI',Arial,sans-serif;background:#f4f5f7;color:#17202b;margin:0;line-height:1.5}
main{max-width:1000px;margin:0 auto;padding:28px 20px 60px;display:grid;gap:18px}
h1{margin:0;font-size:28px} h2{margin:0 0 4px;font-size:20px} h3{margin:12px 0 6px;font-size:15px}
.muted{color:#5b6676;font-size:14px}
.card{background:#fff;border:1px solid #dfe3e9;border-radius:10px;padding:18px;display:grid;gap:10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile{border:1px solid #dfe3e9;border-radius:8px;padding:12px}
.tile b{display:block;font:600 26px ui-monospace,Menlo,monospace}
.up{color:#1d7a4c}.down{color:#b3322a}
table{border-collapse:collapse;width:100%;font-size:14px} td,th{text-align:left;padding:7px 6px;border-bottom:1px solid #dfe3e9;vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#5b6676}
svg{width:100%;height:auto} .grid{stroke:#dfe3e9} .axis{font:11px ui-monospace,monospace;fill:#5b6676}
.mark{stroke:#9a6400;stroke-dasharray:4 4;stroke-width:1.5} .markdot{fill:#fbf0d9;stroke:#9a6400}
.marknum{font:600 11px ui-monospace,monospace;fill:#9a6400} .endlbl{font:600 12px ui-monospace,monospace}
.pill{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px}
.hit{background:#e2f3ea;color:#1d7a4c}.miss{background:#fbe5e3;color:#b3322a}.part{background:#fbf0d9;color:#9a6400}
"""


def build_report(cfg: dict, con: sqlite3.Connection, out: Path) -> Path:
    parts = [f"<header><h1>AI Visibility Report</h1><p class='muted'>Generated {dt.datetime.now():%b %d, %Y %H:%M}. "
             f"Mention rate = share of customer-question answers that name the business "
             f"({cfg['settings'].get('repeats', 3)} runs per question per provider).</p></header>"]
    for c in cfg["clients"]:
        runs = con.execute("SELECT * FROM runs WHERE client_id=? ORDER BY run_at", (c.id,)).fetchall()
        changes = con.execute("SELECT * FROM changes WHERE client_id=? ORDER BY date", (c.id,)).fetchall()
        points: dict[str, list] = {}
        for r in runs:
            points.setdefault(r["provider"], []).append((r["run_at"], r["rate"]))
        tiles, qtables, sources_html = [], [], ""
        for prov in points:
            d = diff_runs(con, c.id, prov)
            lt = d["latest"]
            delta = ""
            if d["prev"]:
                dv = lt["rate"] - d["prev"]["rate"]
                delta = f"<span class='{'up' if dv >= 0 else 'down'}'>{'+' if dv >= 0 else ''}{dv:.0f} pts</span>"
            tiles.append(f"<div class='tile'><span class='muted'>{html.escape(prov)}</span><b>{lt['rate']:.0f}%</b>"
                         f"<span class='muted'>{lt['hits']}/{lt['total']} answers {delta}<br>"
                         f"avg position {lt['avg_position'] or '—'} · brand: {lt['brand_sentiment'] or '—'}</span></div>")
            pq = per_question(con, lt["id"])
            prevq = per_question(con, d["prev"]["id"]) if d["prev"] else {}
            rows = []
            for qi, q in sorted(pq.items()):
                share = q["hits"] / q["n"]
                cls = "hit" if share >= 0.67 else "part" if share > 0 else "miss"
                mv = ""
                if qi in prevq and not q["branded"]:
                    was = prevq[qi]["hits"] / prevq[qi]["n"]
                    mv = "<span class='up'>Gained</span>" if share - was >= 0.5 else "<span class='down'>Lost</span>" if was - share >= 0.5 else ""
                pos = f"#{sum(q['pos'])/len(q['pos']):.1f}" if q["pos"] else "—"
                rows.append(f"<tr><td>{'<i>Brand</i> · ' if q['branded'] else ''}{html.escape(q['question'])}</td>"
                            f"<td><span class='pill {cls}'>{q['hits']}/{q['n']}</span></td><td>{pos}</td><td>{mv}</td></tr>")
            qtables.append(f"<h3>{html.escape(prov)}: latest check, question by question</h3><table><tr><th>Question</th>"
                           f"<th>Named</th><th>Avg position</th><th>Since last</th></tr>{''.join(rows)}</table>")
        if runs:
            latest_ids = [diff_runs(con, c.id, p)["latest"]["id"] for p in points]
            src: dict[str, int] = {}
            sov: dict[str, int] = {}
            for rid in latest_ids:
                for a in con.execute("SELECT sources, businesses, branded FROM answers WHERE run_id=?", (rid,)):
                    for dmn in json.loads(a["sources"] or "[]"):
                        src[dmn] = src.get(dmn, 0) + 1
                    if not a["branded"]:
                        for b in json.loads(a["businesses"] or "[]"):
                            key = next((k for k in sov if same_biz(k, b)), b)
                            sov[key] = sov.get(key, 0) + 1
            top_src = sorted(src.items(), key=lambda kv: -kv[1])[:10]
            top_sov = sorted(sov.items(), key=lambda kv: -kv[1])[:8]
            sources_html = (
                "<h3>Most-named businesses (latest checks, all providers)</h3><table><tr><th>Business</th><th>Answers</th></tr>"
                + "".join(f"<tr><td>{'<b>' if same_biz(n, c.name) else ''}{html.escape(n)}{'</b>' if same_biz(n, c.name) else ''}</td><td>{k}</td></tr>" for n, k in top_sov)
                + "</table><h3>Sites the AI cited</h3><table><tr><th>Site</th><th>Citations</th></tr>"
                + "".join(f"<tr><td>{html.escape(dm)}</td><td>{k}</td></tr>" for dm, k in top_src) + "</table>")
        clog = "".join(f"<tr><td>{i}</td><td>{html.escape(ch['date'])}</td><td>{html.escape(ch['what'])}</td>"
                       f"<td>{html.escape(ch['targets'] or '')}</td></tr>" for i, ch in enumerate(changes, 1))
        parts.append(
            f"<section class='card'><h2>{html.escape(c.name)}</h2><p class='muted'>{html.escape(c.category)} · {html.escape(c.city)}"
            f"{' · ' + html.escape(c.website) if c.website else ''} · {len(runs)} checks</p>"
            f"<div class='tiles'>{''.join(tiles) or '<p class=muted>No checks yet.</p>'}</div>"
            f"{trend_svg(points, changes)}"
            + (f"<h3>Change log (numbered markers on the chart)</h3><table><tr><th>#</th><th>Date</th><th>What changed</th><th>Targets</th></tr>{clog}</table>" if changes else "")
            + "".join(qtables) + sources_html + "</section>")
    out.write_text(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                   f"<title>AI Visibility Report</title><style>{REPORT_CSS}</style></head><body><main>{''.join(parts)}</main></body></html>",
                   encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_run(cfg, con, args):
    providers = build_providers(cfg, args.mock)
    clients = [c for c in cfg["clients"] if not args.client or c.id == args.client]
    if not clients:
        sys.exit(f"No client with id {args.client!r}")
    summary_lines = []
    for c in clients:
        print(f"\n{c.name} ({c.city})")
        for p in providers:
            print(f"  {p.name}: asking {len(c.questions())} questions...")
            res = check_client(c, p, cfg, con, args.mock)
            d = diff_runs(con, c.id, p.name)
            delta = ""
            if d and d["prev"]:
                dv = d["latest"]["rate"] - d["prev"]["rate"]
                delta = f" ({'+' if dv >= 0 else ''}{dv:.0f} pts)"
            line = f"{c.name} · {p.name}: named in {res['hits']}/{res['total']} answers = {res['rate']:.0f}%{delta}"
            print("  " + line)
            for q in (d or {}).get("gained", []):
                print(f"    + gained: {q}")
            for q in (d or {}).get("lost", []):
                print(f"    - lost:   {q}")
            summary_lines.append(line)
            alert_drop = float(cfg["settings"].get("alert_drop_points", 15))
            if d and d["prev"] and d["prev"]["rate"] - d["latest"]["rate"] >= alert_drop:
                summary_lines.append(f"  ALERT: {c.name} dropped {d['prev']['rate'] - d['latest']['rate']:.0f} pts on {p.name}")
    path = build_report(cfg, con, HERE / cfg["settings"].get("report_file", "report.html"))
    print(f"\nReport written to {path}")
    notify(cfg, ["AI visibility check"] + summary_lines)


def cmd_change(cfg, con, args):
    if args.client not in {c.id for c in cfg["clients"]}:
        sys.exit(f"No client with id {args.client!r}")
    date = args.date or dt.date.today().isoformat()
    dt.date.fromisoformat(date)
    con.execute("INSERT INTO changes (client_id, date, what, targets) VALUES (?,?,?,?)", (args.client, date, args.what, args.targets))
    con.commit()
    print(f"Logged change for {args.client} on {date}: {args.what}")


def cmd_status(cfg, con, args):
    for c in cfg["clients"]:
        print(f"{c.name}")
        provs = [r["provider"] for r in con.execute("SELECT DISTINCT provider FROM runs WHERE client_id=?", (c.id,))]
        if not provs:
            print("  no checks yet")
        for p in provs:
            d = diff_runs(con, c.id, p)
            lt = d["latest"]
            print(f"  {p}: {lt['rate']:.0f}% ({lt['hits']}/{lt['total']}) on {lt['run_at'][:10]}"
                  + (f", was {d['prev']['rate']:.0f}%" if d["prev"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Track whether AI assistants recommend your clients.")
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="check clients now")
    r.add_argument("--client", help="only this client id")
    r.add_argument("--mock", action="store_true", help="offline test with fake answers")
    sub.add_parser("report", help="rebuild report.html")
    ch = sub.add_parser("change", help="log a fix you shipped")
    ch.add_argument("client"); ch.add_argument("what")
    ch.add_argument("--date"); ch.add_argument("--targets", default="")
    sub.add_parser("status", help="print latest numbers")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    con = db_connect(HERE / cfg["settings"].get("database", "visibility.db"))
    if args.cmd == "run":
        cmd_run(cfg, con, args)
    elif args.cmd == "report":
        print(f"Report written to {build_report(cfg, con, HERE / cfg['settings'].get('report_file', 'report.html'))}")
    elif args.cmd == "change":
        cmd_change(cfg, con, args)
    elif args.cmd == "status":
        cmd_status(cfg, con, args)


if __name__ == "__main__":
    main()
