# 15 - Engine Summarize & Card

Summarize an engine by tactic and generate a consolidated **engine card**. This
ports the two summary utilities added on the trex `main` / `dev-trt-10.9-update`
branches:

+ `utils/summarize_engine.py` -> a per-tactic latency table (the `trex summary` sub-command).
+ `utils/gen_engine_card.py` -> an engine "report card".

The original engine card is a browser-opening **HTML** page. Following the cookbook
style (text / figures to file, no browser or interactivity), it is produced here
as a **Markdown** file (`engine_card.md`) bundling the summary, the layer-type /
tactic / precision breakdowns and the lint hazards.

`summarize_engine_tactics(plan, group_tactics, sort_key)` returns the per-tactic
`count` / `latency %` rows (see `tensorrt_cookbook/utils_engine_explorer.py`).

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # print the tactic summary + write engine_card.md (no GPU required)
```

## Output

+ `case_summarize_tactics` - a per-tactic latency table (grouped tactics, sorted by latency).
+ `engine_card.md` - a consolidated Markdown report: summary, latency-by-type, top tactics,
  precision bytes, and lint hazards.
