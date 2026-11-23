# AI Visibility Monitor

Checks every week whether ChatGPT, Claude and Perplexity recommend each client business. It saves every answer, tracks the trend, and shows whether the fixes you shipped moved the numbers.

## What it does on each run

1. Asks each AI assistant, with web search on, 8 fixed questions per client: 7 phrased the way customers ask ("best sign company in Columbus, OH") plus one that names the business. Each customer question runs 3 times, because AI answers vary.
2. Reads every answer and records which businesses were named, in what order, whether the client was named, and which websites were cited.
3. Saves everything to `visibility.db` (SQLite).
4. Compares the results with the last check: mention rate, average position, and questions gained or lost.
5. Rebuilds `report.html`: a trend chart per client with your logged fixes marked on it, a question-by-question table, the competitors named most, and the sites the AI cited.
6. Optionally posts a summary to Slack or Discord, with an alert if a client's rate drops by 15 points or more.

## Setup (about 10 minutes)

Requires Python 3.11 or newer. There's nothing to install.

```bash
cp config.example.toml config.toml   # add your clients here
export OPENAI_API_KEY=sk-...          # enable whichever providers you use
export ANTHROPIC_API_KEY=sk-ant-...
python3 monitor.py run --mock         # offline test with fake answers, no keys needed
python3 monitor.py run                # real check
open report.html
```

Put keys in a `.env` file next to `run_weekly.sh` if you run it from cron. Never commit `.env` or `config.toml` with keys in it.

## Daily use

| Command | What it does |
| --- | --- |
| `python3 monitor.py run` | Check all clients now and rebuild the report |
| `python3 monitor.py run --client brightline` | Check one client |
| `python3 monitor.py change brightline "Claimed Bing Places" --targets "All questions"` | Log a fix (shows as a numbered marker on the chart) |
| `python3 monitor.py status` | Print each client's latest rate and the previous one |
| `python3 monitor.py report` | Rebuild the report without running checks |

## Running it automatically every week

- **Mac or Linux:** `crontab -e`, then add
  `40 7 * * 1 /full/path/to/run_weekly.sh >> /full/path/to/monitor.log 2>&1`
  This runs every Monday at 7:40 AM. The computer must be on at that time.
- **Windows:** in Task Scheduler, create a weekly task that runs `python monitor.py run` in this folder.
- **No computer needed:** push this folder to a private GitHub repo and add the API keys as repository secrets. `.github/workflows/weekly-check.yml` then runs it every Monday and commits the updated database and report.

## How to read the numbers

- **Mention rate:** the share of customer-question answers that name the client. 7 questions × 3 runs = 21 answers per provider, plus any extra questions.
- **Trust trends, not single jumps.** A change that holds for 2–3 weekly checks is real. A one-week jump can be noise.
- **Gained or lost:** a question counts as gained when it's named in at least half more of its runs than at the last check.
- **Keep the questions fixed.** Changing a client's name, city or questions resets comparability. Add new questions under `extra_questions` instead.

## Costs and limits

- Each provider uses about 25 web-search calls per client per week, plus one extraction call. Check each provider's current pricing.
- API answers are close to what people see in the consumer apps, but not identical. Location, account history and app features differ.
- API formats change. If a provider starts failing, its parsing code is in one short class in `monitor.py` (`OpenAIProvider`, `AnthropicProvider`, `PerplexityProvider`).
- Model names are set in `config.toml`. Update them as providers release new models.

## Dashboard (`dashboard/index.html`)

The web dashboard used in the demo: clients, mention-rate trend with logged fixes marked on the chart, question-by-question movement, share of voice and cited sources. It runs as a Claude artifact, where it saves data and asks Claude through Claude's page capabilities. Opened as a plain file it shows the built-in example client only.
