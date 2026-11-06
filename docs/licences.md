# Licences (checked 2026-11-05)

Rule: the licence of every code repository, model and dataset is read at its source and recorded here before first use. Fingerprints are the first 16 hex characters of the SHA-256 hash of the licence or model-card file read on 2026-11-05.

| Item | Used for | Version or checkpoint | Licence | Where read | Fingerprint |
|---|---|---|---|---|---|
| `galilai-group/prh-proof` | paper code, tests and result tables under test | commit `a2b9949fa907d4fbe562742d4b1ffa7205ab7427` | MIT, Copyright (c) 2026 Randall Balestriero and Yann LeCun | `LICENSE` in upstream repo at pinned commit | `824b459e9de14e9d` |
| `galilai-group/stable-pretraining` | training library imported by `prh-proof` tests | commit `ab836bf699a2be712dfcf980a5eb70d36391e876` (and `main`) | MIT, Copyright (c) 2024 rbalestr-lab | `LICENSE` in upstream repo | `8fa02eaa4e67a545` |
| WIT-1024 (`minhuh/prh`) | Phase 2 evaluation benchmark (1,024 Wikipedia image-caption pairs from Huh et al. 2024) | Hugging Face dataset `minhuh/prh`, revision `wit_1024` | CC BY-SA 3.0 Unported (upstream `google-research-datasets/wit` licence; Wikipedia texts and Wikimedia images) | upstream `LICENSE` in `google-research-datasets/wit` and dataset card in `minhuh/prh` (`wit_1024`) | `3f941b3b89cf7b83` (`2d9a06017e2e3fdd` card) |
| DINOv2 ViT-B/14 (`dinov2`) | Phase 2 reference image encoder | `timm` `vit_base_patch14_dinov2.lvd142m` (`facebook/dinov2-base`) | Apache License 2.0 | `LICENSE` in `facebookresearch/dinov2` | `600cc67cc4cb2f5e` |
| Qwen2.5-0.5B (`qwen0.5b`) | Phase 2 language model baseline | `Qwen/Qwen2.5-0.5B` | Apache License 2.0 | `LICENSE` in `Qwen/Qwen2.5-0.5B` on Hugging Face | `832dd9e00a68dd83` |
| Qwen2.5-1.5B (`qwen1.5b`) | Phase 2 language model baseline | `Qwen/Qwen2.5-1.5B` | Apache License 2.0 | `LICENSE` in `Qwen/Qwen2.5-1.5B` on Hugging Face | `832dd9e00a68dd83` |
| Qwen2.5-3B (`qwen3b`) | Phase 2 language model baseline | `Qwen/Qwen2.5-3B` | Qwen Research License Agreement (permits non-commercial research use) | `LICENSE` in `Qwen/Qwen2.5-3B` on Hugging Face | `ef52482bb7857330` |
| Qwen2.5-7B (`qwen7b`) | Phase 2 language model baseline | `Qwen/Qwen2.5-7B` | Apache License 2.0 | `LICENSE` in `Qwen/Qwen2.5-7B` on Hugging Face | `832dd9e00a68dd83` |
| BERT-base (`bert`) | Phase 2 bidirectional text baseline | `google-bert/bert-base-uncased` | Apache License 2.0 | `README.md` frontmatter in `google-bert/bert-base-uncased` on Hugging Face | `9187b6018ea0010d` |
| all-mpnet-base-v2 (`mpnet`) | Phase 2 contrastive sentence encoder baseline | `sentence-transformers/all-mpnet-base-v2` | Apache License 2.0 | `README.md` frontmatter in `sentence-transformers/all-mpnet-base-v2` on Hugging Face | `89a1a9c3290fe58e` |
| CLIP ViT-B/32 text (`clip`) | Phase 2 image-text contrastive baseline | `openai/clip-vit-base-patch32` | MIT, Copyright (c) 2021 OpenAI | `LICENSE` in `openai/CLIP` and `README.md` in `openai/clip-vit-base-patch32` | `987e63b32f6c89ff` (`8f0e8c678e0afe5f` card) |
