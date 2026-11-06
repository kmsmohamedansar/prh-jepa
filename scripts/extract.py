"""Extract per-layer features on WIT-1024 for one frozen vision or text encoder.

Follows the PRH benchmark protocol (Huh et al. 2024; Balestriero and LeCun 2026):
- Vision encoder (dinov2): timm vit_base_patch14_dinov2.lvd142m at native
  resolution (518x518), CLS token after each of the 12 transformer blocks.
- Text encoders: first caption of each WIT-1024 pair, tokenized up to 512
  tokens (77 for CLIP), attention-masked mean of every hidden state.
"""
import argparse, json, pathlib, platform, time
import numpy as np

IMAGE_MODELS = {
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
}

TEXT_MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen3b": "Qwen/Qwen2.5-3B",
    "qwen7b": "Qwen/Qwen2.5-7B",
    "bert": "google-bert/bert-base-uncased",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "clip": "openai/clip-vit-base-patch32",
}

ALL_MODELS = {**IMAGE_MODELS, **TEXT_MODELS}
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _masked_mean(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)


def extract_vision(model_key, images, device, batch_size):
    import timm, torch

    name = IMAGE_MODELS[model_key]
    net = timm.create_model(name, pretrained=True, num_classes=0)
    cfg = timm.data.resolve_data_config(net.pretrained_cfg, model=net)
    tf = timm.data.create_transform(**cfg)
    net = net.eval().to(device, torch.float32)
    pool = (lambda o: o[:, 0]) if getattr(net, "num_prefix_tokens", 1) else (lambda o: o.mean(1))
    feats = []
    hooks = [blk.register_forward_hook(lambda m, inp, out: feats.append(pool(out).float().cpu())) for blk in net.blocks]
    with torch.no_grad():
        for chunk in _batches(list(images), batch_size):
            x = torch.stack([tf(im.convert("RGB")) for im in chunk]).to(device, torch.float32)
            net(x)
    for h in hooks:
        h.remove()
    n_blocks = len(net.blocks)
    layers = [torch.cat(feats[i::n_blocks]).numpy().astype(np.float32) for i in range(n_blocks)]
    return layers, "float32"


def extract_text(model_key, texts, device, batch_size):
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = TEXT_MODELS[model_key]
    use_bf16 = model_key in ("qwen3b", "qwen7b")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    net = AutoModel.from_pretrained(name, dtype=dtype).eval().to(device)
    max_len = 77 if model_key == "clip" else 512
    chunk_layers = []
    with torch.no_grad():
        for chunk in _batches(list(texts), batch_size):
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            if model_key == "clip":
                hs = net.text_model(**enc, output_hidden_states=True).hidden_states
            else:
                hs = net(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"]
            chunk_layers.append([_masked_mean(h, mask).cpu().numpy().astype(np.float32) for h in hs])
    layers = [np.concatenate([c[i] for c in chunk_layers], axis=0) for i in range(len(chunk_layers[0]))]
    return layers, "bfloat16" if use_bf16 else "float32"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(ALL_MODELS), required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="mps or cpu (default: mps if available)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / "results" / "features"
    raw_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wit1024-{args.model}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} already exists; pass --overwrite to recompute")
        return

    import datasets, timm, torch, transformers

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    ds = datasets.load_dataset("minhuh/prh", revision="wit_1024", split="train")
    if args.model in IMAGE_MODELS:
        layers, dtype_str = extract_vision(args.model, ds["image"], device, args.batch_size)
    else:
        texts = [t[0] for t in ds["text"]]
        layers, dtype_str = extract_text(args.model, texts, device, args.batch_size)
    elapsed = round(time.time() - t0, 2)

    np.savez(out_path, **{f"layer_{i}": arr for i, arr in enumerate(layers)})
    meta = {
        "benchmark": "WIT-1024 (minhuh/prh, revision wit_1024, split train)",
        "model_key": args.model,
        "checkpoint": ALL_MODELS[args.model],
        "modality": "image" if args.model in IMAGE_MODELS else "text",
        "device": device,
        "dtype": dtype_str,
        "batch_size": args.batch_size,
        "n_items": int(layers[0].shape[0]),
        "n_layers": len(layers),
        "dim": int(layers[0].shape[1]),
        "seconds": elapsed,
        "machine": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "timm": timm.__version__,
            "datasets": datasets.__version__,
            "numpy": np.__version__,
        },
    }
    (raw_dir / f"extract-{args.model}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
