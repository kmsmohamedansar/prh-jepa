"""Save features for one model on one split. Not yet run on real models."""
import argparse, json, pathlib
import numpy as np

MODELS = {  # checkpoints to confirm in docs/licences.md before first use
    "ijepa": "facebook/ijepa_vith14_1k",
    "mae": "facebook/vit-mae-huge",
    "dinov2": "facebook/dinov2-large",
    "clip": "openai/clip-vit-large-patch14",
    "lm": None,  # language model: to be chosen
}
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--split", choices=["dev", "heldout"], required=True)
    p.add_argument("--run-held-out-once", action="store_true")
    args = p.parse_args()
    if args.split == "heldout" and not args.run_held_out_once:
        raise SystemExit("held-out is scored once: pass --run-held-out-once")
    out = ROOT / "results" / "features" / f"{args.split}-{args.model}.npy"
    if args.split == "heldout" and out.exists():
        raise SystemExit(f"{out} exists; held-out is never re-run")
    if MODELS[args.model] is None:
        raise SystemExit("model not chosen yet; see HANDOFF.md")
    import torch
    from transformers import AutoModel, AutoProcessor
    # TODO: load the frozen dataset split, run the model, mean-pool the last
    # hidden state, save features plus a JSON record of versions and machine.
    raise SystemExit("dataset not chosen yet; see HANDOFF.md")


if __name__ == "__main__":
    main()
