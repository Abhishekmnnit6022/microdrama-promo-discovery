# Output analysis: full-dataset reference run

Run: `python run_pipeline.py --out raw_outputs --llm`, 2026-11-20 16:11–16:13, pipeline v3.0.0.
Every number below comes from the files in this folder: `5_run_logs/run_manifest.json`, `6_evaluation_and_tests/run_breakdown.txt`,
`6_evaluation_and_tests/evaluation_summary.json`, `2_other_candidates/candidates_all.csv`, `1_recommendations/recommendations.json`, `5_run_logs/performance.json`.
None of the generated files were edited by hand.

**What each kind of evidence is.**
*Automated* = computed by code from the run.
*Silver proxy* = derived from script structure. This is not human judgement.
*Gemini* = an LLM's opinion (`gemini-3.5-flash-lite`; the primary `gemini-3.8-flash` returned 503/429 errors and
the code fell back automatically, see `5_run_logs/run_log.txt`).
*Claude review* = my own reading of the outputs while building the system. That is an AI's reading, **not editorial
validation**.
**No human editor has reviewed any recommendation yet**, so editor acceptance, hook/cliffhanger precision and
recall@K are **not measured**.

## 1. Episodes processed

50 episode scripts discovered in one show (`data/I Married a Broken Soldier. He was a Duke/`).
50 parsed and processed, **0 failed**, 0 parse warnings (`5_run_logs/errors.jsonl` is empty).
2 episodes (40 and 41) produced no valid single-episode promo. Their reasons are in section 8.

## 2. Candidates generated

**25,771 candidates**:

* 15,790 continuous
* 1,649 multi-segment (same episode)
* 8,332 cross-episode (episode N + N+1/N+2)

In addition, 3,871 hook/ending pairs were discarded before scoring because the two segments shared no
character (logged in `5_run_logs/run_log.txt`). From these, 1,550 candidates (the top 25 per episode plus the top 300
multi/cross) went to stage 2 (zero-shot NLI re-scoring).

## 3. Duration validation

25,771 / 25,771 candidates lie in 20–60 s (100 %; the generator only emits valid lengths, and this is re-checked).
The 15 recommended promos run **20.6 s – 57.9 s**, mean 38.5 s. `6_evaluation_and_tests/validation.json` re-checks each one:
segment bounds equal source line timestamps, no untimed lines, story order, hook inside the first 3 s, cliffhanger
timestamp equal to the cut, and no overlapping pairs. **15/15 pass**, with 0 duplicate or overlapping pairs.

## 4. Valid hooks

14,621 / 25,771 candidates (56.7 %) pass the hook gate (≥ 0.45).
"Weak opening hook" is the most common rejection reason (11,150 candidates).

## 5. Valid cliffhangers

18,925 / 25,771 (73.4 %) pass the cliffhanger gate (score ≥ 0.50 and open-loop signal ≥ 0.50); 6,846 were rejected.
Counts mix stage-1 scores (most candidates) with stage-2 scores (the 1,550 shortlisted). Within the shortlist,
227 failed the cliffhanger gate and 249 the hook gate.

## 6. Rejected for poor coherence

**Only 155 candidates (0.6 %)** fall below the coherence gate (0.55). This gate rarely decides anything, and the
evidence in section 8 suggests it is **too lenient**.

All gates passed: 12,096 candidates (46.9 %). Stage-2 shortlist: 1,141 / 1,550 passed.
After diversity selection (≤ 2 per episode, ≤ 30 % overlap, no shared hook or cut line), 15 were recommended.

## 7. Strongest recommendations

Full detail is in `1_recommendations/recommendations_report.md`.

| # | Score | Dur. | Structure | Cut | Hook → Cliffhanger | Gemini (hook / cliff / standalone) |
|---|---|---|---|---|---|---|
| 1 | 0.915 | 57.9 s | continuous | ep 28 01:10.80–02:08.70 | Catherine yanks Juliana's locket → "This book of weeds will hang her before the magistrate." | yes / yes / yes |
| 2 | 0.913 | 54.9 s | continuous | ep 28 00:15.30–01:10.20 | "I did not poison her!" → "Take her away before she poisons the rest of this house." | yes / yes / yes |
| 4 | 0.901 | 24.9 s | continuous | ep 27 02:03.10–02:28.00 | "She admits she has been giving the Duchess poison in secret!" → guard drags Juliana toward the door | yes / yes / yes |
| 7 | 0.890 | 55.2 s | continuous | ep 35 01:29.80–02:25.00 | "A traitor?" → "Who?" (charged question) | yes / yes / yes |
| 8 | 0.879 | 41.9 s | continuous | ep 15 00:07.60–00:49.50 | Juliana bursts through the cottage doorway → thug shoves her back | yes / yes / yes |

Structure mix of the 15: 14 continuous, 1 cross-episode (#13, ep 24 → 25), and no multi-segment.
The best multi-segment promos are listed separately in `2_other_candidates/alternatives_by_structure.json` (the top one, ep 28 + ep 28, scores 0.879).
Episodes covered: 3, 15, 21, 24, 25, 26, 27, 28, 29, 30, 35, 36, 43, 47.
Gemini's overall agreement: hook 15/15, cliffhanger 13/15, standalone 10/15.

## 8. Where the system performed badly

**Poor recommendations that made the top 15, and why they were selected**

* **#12 (ep 21 00:11.20–00:40.40) ends on resolution.** It ends with "Stand up… Let me grab my ledger. We go now."
  The zero-shot model gave the closing line *threat 0.97*, and that alone supplied the open-loop signal. Gemini:
  cliffhanger = no ("ends on a resolution"). An earlier run's version of the same failure (the physical cue
  "grab") was fixed. The zero-shot path re-introduced it.
* **#3 (ep 3 01:00.50–01:43.80) cuts after the rescue is complete.** The final line is "…when your family is
  dragging a defenseless woman from a church." Zero-shot reads it as a threat (0.93), but the scene is already
  resolved. Gemini: cliffhanger = no.
* **#13 (cross-episode, ep 24 → 25) joins a riding-crop strike to an unrelated argument.** It is linked only by the
  shared character Juliana: the noblewoman's strike in ep 24, then Alexander's "you never told me" about a
  poisoning in ep 25. The character-overlap rule is too weak to guarantee that the two segments form one story.
  It was flagged (`risk_flags`), and Gemini judged it standalone = no.
* **#14, #15, #5, #9 need backstory.** Gemini: standalone = no ("who are Juliana, Rupert and Catherine… why
  Alexander lost his home", "what law is being used against her"). Our coherence scores for these were 0.74–0.96,
  so the coherence analysis is too generous. It rewards named speakers, stakes words and logline overlap, which a
  late-season scene full of names easily satisfies, even when a new viewer cannot follow it.

**Episodes with no valid single-episode promo**

* ep 40: 258 / 359 candidates had a weak opening hook and 151 had no genuine cliffhanger.
* ep 41: 288 / 328 weak hook, 117 no cliffhanger.

These are quieter episodes (20 and 18 dialogue lines, mostly ACTION/CAMERA). The system reports this rather than
forcing a recommendation.

**Near misses** (`2_other_candidates/near_misses.json`): the five highest-scoring rejected candidates all failed only the cliffhanger
gate, with an open-loop signal of 0.42–0.50, just under the 0.50 threshold.

## 9. Common failure patterns

1. **Zero-shot "threat" is over-assigned.** 13 of the 15 recommended cuts carry the cliffhanger label
   "zero-shot reads an unresolved threat". Across the 1,550 shortlisted candidates, the leading cliffhanger reasons
   are unanswered question (420), charged statement left hanging (348) and zero-shot threat (329). The small NLI
   model (`deberta-v3-xsmall`) scores many tense-but-closed lines as threats, which caused #3 and #12.
2. **Coherence is too lenient.** Only 0.6 % of candidates are rejected for coherence, yet Gemini found 5/15 of the
   final promos hard to follow without backstory. The rules check surface signals (names, stakes words) rather than
   whether the *situation* is introduced inside the clip.
3. **The cliffhanger detector agrees only weakly with the writers' own endings.** Silver proxy
   (`6_evaluation_and_tests/evaluation_summary.json`): treating each episode's final dialogue line as a known cliffhanger and comparing it
   with 8 other cut points in the same episode gives AUC **0.50** with stage-1 features and **0.54** with zero-shot.
   That is barely above chance. In the first reference run (`7_earlier_runs/v3_run1_before_open_loop_fix/`) it was 0.44,
   because declarative threats ("Let them see what happens when a woman forgets her place.") carried no lexicon
   cue. The fix (zero-shot + charged-statement signal) helped only modestly and caused pattern 1.
   Caveat: many episode endings in this show are soft lines followed by a visual "cliffhanger hold" in a CAMERA
   line, so the proxy itself is imperfect.
4. **Continuous clips dominate.** Multi-segment and cross-episode candidates pass the gates far more often (84 % and
   88 % vs 22 % for continuous), because they are built from pre-selected strong hook beats and strong endings.
   Their continuity component (0.8 / 0.6 vs 1.0) still ranks them below the best continuous cuts. Whether editors
   would prefer them is untested.
5. **The emotion model and the writers' tags agree only weakly.** Pearson r = 0.24 between model arousal and the
   intensity of the script's own emotion tags (1,478 dialogue lines). Both signals are kept, and neither is
   trusted alone.

## 10. Improvements still needed

* Collect editor decisions in the Quality review screen, and a small labelled set in `eval/labels/` (schema
  provided). Then measure acceptance, hook/cliffhanger precision and recall@K, and fit the weights and gates to
  that data instead of hand-setting them.
* Replace the xsmall zero-shot model with a larger NLI model or a cross-check, so that a high "threat" score alone
  cannot pass the cliffhanger gate. Require agreement from a second signal.
* Coherence: test whether the clip itself introduces the conflict (who wants what from whom) instead of counting
  names. Use the Gemini standalone verdict as an optional gate, not just a note.
* Cross-episode: require a shared *plot thread* (shared non-name content words, or the episode continuity notes),
  not only a shared character.
* Hindi/Hinglish lexicons and a multilingual emotion model (none needed for this dataset).

## Performance (measured, not estimated)

Machine: Windows 11, CPU only (see `5_run_logs/performance.json` → `machine`).

| Run | Total | Per episode | Bottleneck |
|---|---|---|---|
| Cold caches, no LLM (`7_earlier_runs/v3_cold_cache_benchmark/`, one revision earlier) | 609.6 s | 12.2 s | zero-shot NLI 584.6 s (96 %) for 1,041 boundary lines; emotion model 18.3 s for 1,478 lines |
| This reference run, warm caches + Gemini | 102.2 s | 2.0 s | Gemini review 62.8 s; zero-shot 34.6 s (new boundary lines only) |

Candidate generation and scoring of all 25,771 candidates takes about 4 s. Only new episodes and new boundary lines
cost model time in a weekly run. The run manifest reports `emotion: not loaded` because every line was served from
`.cache/`. The status message now says so explicitly; that fix was made after this run.

## Test results

`6_evaluation_and_tests/pytest_output.txt`: **77 passed**. Tests cover parsing (TXT/SRT/JSON, malformed and missing timestamps,
episode IDs, Devanagari text), duration limits, the hook/cliffhanger/coherence detectors on controlled scenes,
multi-segment and cross-episode construction, gates, diversity, exports, the end-to-end pipeline with a broken file,
cache resume, model failure and LLM failure. The tests do not load the ML models; the models are exercised only by the full runs above.
