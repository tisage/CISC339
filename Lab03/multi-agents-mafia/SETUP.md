# Setup: Generating Your Own Mafia Game

This is for students who want to generate a **new** game with their own
OpenRouter API key, instead of only replaying the pre-generated games that
ship with `Lab_03_Mafia_Demo.ipynb`. The notebook itself never asks for or
touches an API key - all of that lives here.

## ⚠️ API key safety - read this first

Your OpenRouter API key is tied to **your own billing**. Anyone who gets a
copy of it can spend your money. Some concrete rules:

- **Never paste your key into a notebook cell, a chat message, a screenshot,
  or anything you might share, submit, or push to GitHub.** A notebook you
  hand in for a homework submission is exactly the kind of place a key
  accidentally ends up committed forever in someone's git history.
- **Never commit a `.env` file.** This repo's `.gitignore` already excludes
  `.env`, but double-check with `git status` before you `git add` anything
  in this folder.
- **Set a spending limit** on your OpenRouter account (Settings → Limits) so
  a bug in your own code (e.g. an accidental infinite loop calling the API)
  can't run up an unexpectedly large bill.
- **If you ever suspect your key leaked** (e.g. you accidentally pasted it
  somewhere public), revoke it immediately on openrouter.ai and generate a
  new one - don't wait to see if anything bad happens first.
- The scripts in this folder read the key from a `.env` file via
  `OPENROUTER_API_KEY`, never from a hardcoded string in the `.py` files -
  keep it that way if you modify anything.

## Steps

1. Create an account at [openrouter.ai](https://openrouter.ai) and generate
   an API key under your account settings.
2. (Recommended) Set a spending limit on the key before you do anything
   else.
3. In this folder (`Lab03/multi-agents-mafia/`), create a file named `.env`
   (copy `.env.example` as a starting point) containing:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```
4. Create the project's virtual environment and install dependencies (from
   this folder):
   ```bash
   uv venv .venv --python 3.12
   uv pip install -p .venv -r requirements.txt
   ```
5. Verify all 6 roster models are reachable before spending money on a full
   game - this is a handful of tiny, cheap requests, not a full game:
   ```bash
   .venv/bin/python3 test_models.py
   ```
   If any model fails, fix it (or ask the instructor) before continuing.
6. Generate one game:
   ```bash
   .venv/bin/python3 generate_games.py --n 1
   ```
   This costs roughly $0.15-$0.80 depending on how many rounds the game
   runs (cost grows with round count, since each agent's prompt includes
   the full game history so far). Watch the printed running-total cost.
7. A new file appears in `games/`. Open `Lab_03_Mafia_Demo.ipynb`, re-run
   the game-picker cell, and your new game will show up as a dropdown
   option automatically.

## Cost guardrails

`generate_games.py` supports flags to keep a batch run from running away:

```bash
.venv/bin/python3 generate_games.py --n 5 --budget-usd 3.0   # stop once $3 is spent
.venv/bin/python3 generate_games.py --n 1 --seed-start 42     # reproducible seed
```

## Optional: compare private reasoning to public statements

After generating your own game, open the resulting JSON file directly (it's
plain text - any editor works) and look at the `day_discussion` entries in
each round. Each one has both a `public_statement` (what other players see)
and a `private_reasoning` (what only that agent "thinks," never shown to
anyone else in-game). Compare the two for the same agent in the same round -
where do they diverge? That gap is the AI actively managing what it reveals,
not just being random.
