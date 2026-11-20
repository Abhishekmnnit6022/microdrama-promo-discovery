"""Command-line batch run over a whole dataset folder.

Examples (from the project folder):
    python run_pipeline.py                                  # data/ -> outputs/latest, no LLM
    python run_pipeline.py --out raw_outputs --llm          # reference run with Gemini review of the final list
    python run_pipeline.py --dataset path/to/shows --top-k 20 --no-zeroshot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.batch import RunConfig, output_path, run, write_outputs  # noqa: E402
from src.llm_review import resolve_api_key  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AI microdrama promo discovery — batch run")
    ap.add_argument("--dataset", default=str(ROOT / "data"), help="folder containing one sub-folder per show")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "latest"), help="output folder")
    ap.add_argument("--cache", default=str(ROOT / ".cache"), help="parse/inference cache folder")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-per-episode", type=int, default=2)
    ap.add_argument("--mix", choices=["best", "include_multi_episode", "multi_episode_only"], default="best",
                    help="which promos fill the final list; scores are unchanged")
    ap.add_argument("--multi-episode-slots", type=int, default=5, help="slots reserved by --mix include_multi_episode")
    ap.add_argument("--max-episodes", type=int, default=None, help="limit episodes per show (debugging)")
    ap.add_argument("--no-emotion", action="store_true", help="skip the emotion model (script tags only)")
    ap.add_argument("--no-zeroshot", action="store_true", help="skip the zero-shot NLI stage")
    ap.add_argument("--llm", action="store_true", help="Gemini second opinion on the final list (needs GEMINI_API_KEY)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    log_file = output_path(out, "run_log.txt")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers[:] = [fh, sh]
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = RunConfig(
        dataset_dir=args.dataset, output_dir=str(out), cache_dir=args.cache, top_k=args.top_k,
        max_per_episode=args.max_per_episode, mix=args.mix, multi_episode_slots=args.multi_episode_slots, use_emotion=not args.no_emotion, use_zeroshot=not args.no_zeroshot,
        use_llm=args.llm, api_key=resolve_api_key(env_file=ROOT / ".env") if args.llm else "",
        max_episodes=args.max_episodes,
    )
    res = run(cfg, progress=lambda p, m: logging.getLogger("promo.progress").info("%3.0f%% %s", p * 100, m))
    if not res.shows:
        logging.error("Nothing processed: %s", res.stats.get("error"))
        return 1
    files = write_outputs(res, out)
    logging.info("Wrote %d files to %s", len(files), out)
    s = res.stats["summary"]
    print("\nSUMMARY")
    for k, v in s.items():
        print(f"  {k:32s} {v}")
    print(f"  {'validation':32s} {res.stats['validation']['promos_valid']}/{res.stats['validation']['promos_checked']} recommended promos pass automated checks")
    return 0 if s["episodes_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
