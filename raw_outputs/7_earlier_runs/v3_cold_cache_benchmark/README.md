# Cold-cache benchmark run (unedited)

Full 50-episode run made with an empty `.cache/` (no cached emotion / zero-shot outputs; models already
downloaded), on 2026-11-20 at 15:54–16:04, without the Gemini review.

It was made with the pipeline *one revision before* the reference run in `raw_outputs/` (folders 1–6). Two lexicon fixes came
after it (spoken physical verbs down-weighted; de-escalating ACTION lines treated as resolution, and "the duke"
and "heir" removed from the revelation cues). So its rankings differ slightly from the reference run. It is kept
because it is the only run that measures model inference time from scratch.
Files: `run_log.txt`, `performance.json`, `run_manifest.json`, copied as written by the pipeline.
