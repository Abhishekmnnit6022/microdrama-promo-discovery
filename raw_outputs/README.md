# Raw outputs: full-dataset run

Unedited output of running the solution on the **full input pack**: 1 show, all 50 episodes
(`kuku_fm_assignment_project/data/I Married a Broken Soldier. He was a Duke/`).

Produced by: `python run_pipeline.py --out raw_outputs --llm` (run from the repository root), 2026-11-20 16:11.
The files in folders 1–6 were written by the pipeline, the evaluation script and pytest. They were only sorted into folders, never edited.
Only this README and `OUTPUT_ANALYSIS.md` are hand-written.

## Start here

| If you want to… | Open |
|---|---|
| See the final answer: 15 ranked promos with timestamps | `1_recommendations/recommendations_report.md` |
| Read what worked, what failed, and why | `OUTPUT_ANALYSIS.md` |
| Check the run processed every episode | `5_run_logs/run_manifest.json` → `stats.summary` |

## Folder guide

| Folder | What is inside |
|---|---|
| `1_recommendations/` | **The final answer.** 15 ranked promos: episodes, ordered segments with exact start/end timestamps, hook and cliffhanger timestamps, score breakdown, one-line reason, Gemini second opinion. `.json` (full detail), `.csv` (one row per segment), `_report.md` (readable). |
| `2_other_candidates/` | Everything that was not picked. `alternatives_by_structure.json`: best continuous / multi-segment / cross-episode promos. `episode_best.json`: best promo of each episode. `near_misses.json`: high scorers that failed one quality check, with the reason. `candidates_all.csv`: all 25,771 scored candidates. |
| `3_per_episode/` | One file per episode (50): parsing result, line counts, number of candidates, best promo, or why there is none. |
| `4_model_outputs/` | `line_features.csv`: emotion-model and zero-shot scores and keyword cues for every line. `llm_review.json`: Gemini's verdict on each final promo. |
| `5_run_logs/` | `run_log.txt`: full log. `run_manifest.json`: settings, models used, counts, status of every episode. `performance.json`: time per stage. `errors.jsonl`: errors and warnings (empty, since there were none). |
| `6_evaluation_and_tests/` | `validation.json`: automated checks of each final promo (duration, timestamps, order, overlap). `evaluation_summary.json`: evaluation metrics. `run_breakdown.txt`: numbers used in the analysis. `pytest_output.txt`: test results (77 passed). |
| `7_earlier_runs/` | Kept for transparency, **not** the final result. `v2_previous_version/`: outputs of the old implementation. `v3_run1_before_open_loop_fix/`: first full run, before a cliffhanger fix. `v3_cold_cache_benchmark/`: timing of a run with nothing cached. Each has its own short note. |

## Key numbers

| | |
|---|---|
| Episodes processed | 50 / 50, 0 failed |
| Candidate promos generated (all 20–60 s) | 25,771 |
| Passed every quality check | 12,096 |
| Recommended | 15 (20.6–57.9 s), all pass the automated checks |
| Gemini agreed (hook / cliffhanger / standalone) | 15 / 13 / 10 of 15 |
| Run time | 610 s with nothing cached, 102 s with caches + Gemini |
| Human editor review | not done yet |
