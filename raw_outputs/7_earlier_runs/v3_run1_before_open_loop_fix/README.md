# Iteration 1: first reference run (unedited)

Full 50-episode run made 2026-11-20 16:05–16:07 with Gemini review. It is kept as-is because analysing it led
to a change:

* Silver cliffhanger proxy (eval/evaluate.py, first version, stage-1 features only): AUC 0.436. The detector
  preferred other cut points over the writers' episode-ending lines. Reason: the open-loop signal only came from
  lexicon cues, while episode endings are mostly declarative threats or revelations with no cue words.
* Gemini judged 4/15 recommendations standalone.

Change made afterwards (src/cliffhanger.py): zero-shot threat/revelation/suspense and "charged statement left
hanging" (high emotion-model arousal or a high-intensity script tag) can now supply the open-loop signal.
The current reference run in `raw_outputs/` (folders 1–6) uses the changed code.
