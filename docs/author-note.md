# Courtesy note to the `prh-proof` authors (draft)

Per rule 4 in `CLAUDE.md`, share this note (by email or on `github.com/galilai-group/prh-proof`) so the authors have a chance to reply before publishing on `ansarbuilds.dev`.

---

**Subject:** Independent laptop-scale replication & audit of `galilai-group/prh-proof` (`arXiv:2610.39302`)

Dear Randall Balestriero and Yann LeCun,

I recently ran a small, independent replication and code/table audit of *Proving and Improving the Platonic Representation Hypothesis with JEPAs* (`galilai-group/prh-proof` at commit `a2b9949fa907`, with `stable-pretraining` at `ab836bf699a2`) on an Apple M4 MacBook Air (16 GB unified memory), before posting a technical write-up on my personal site. Before publishing anything, I wanted to share what I found and give you a chance to flag anything I may have misread.

### What held up cleanly

1. **Test suite (`pytest`):** 135 of 139 tests passed out of the box on CPU (`133` in `-m "not slow"`, `2` in `-m "slow"`), and 3 skipped cleanly for optional extras (`decord`, `pycocoevalcap`, `unpaired_rosetta`).
2. **Independent WIT-1024 baselines:** Using separate NumPy implementations of mutual 10-NN, unbiased linear CKA and `prh_prep`, I re-extracted `DINOv2-B` and six open text models (`Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`, `bert-base-uncased`, `all-mpnet-base-v2`, `clip-vit-base-patch32`) on `minhuh/prh` (`wit_1024`). All six matched your `llm_scale.csv` and `wit_cka.csv` tables within `0.0012` on mutual 10-NN and `0.0004` on CKA.
3. **Finite worlds and small LeJEPA check:** Your `prh.theory` checks reproduced cleanly across 20 clustered product worlds and sample-density sweeps. In a tiny 3-seed `BERT-T` LeJEPA run on `COCO Karpathy` captions (`51,200` scenes = `204,800` views per run), discrete mutual 10-NN on WIT-1024 was still near initialization (`0.0305` bad views vs `0.0317` good views, which my report explicitly treats as a laptop scale limit), while global unbiased CKA separated good views from bad views on all 3 seeds (`0.0636` vs `0.0439`).

### Five minor notes and questions I plan to mention

1. **`pytest 9.2.0` compatibility (`tests/test_metrics.py`):** `test_unbiased_cka_matches_the_u_statistic_and_its_invariances` raises `TypeError: iteration over a 0-d tensor` under `pytest 9.2.0` because `pytest.approx(reference)` checks `isinstance(expected, Iterable)` on a 0-d `torch.Tensor` (the underlying CKA values agree to `< 1e-16` once `reference.item()` or `float(reference)` is passed).
2. **PyTorch MPS deadlock in `SlicedEppsPulley`:** In `stable_pretraining/losses/lejepa.py:125`, `torch.Generator(device=x.device).manual_seed(...)` hangs on Apple MPS in PyTorch `2.15.0`; running that scalar projection test on CPU fixed it cleanly.
3. **Sample ratio vs token ratio on the "42x fewer samples" headline:** From `scaling.csv` and `phase_real.csv`, `BERT-S` good views (`21.6M` samples, `.108` WIT mean) vs `BERT-B` next-token (`900M` samples, `.111` mean) is `41.67x` in samples and `13.89x` in tokens (`38.52B / 2.77B`, since each good-view sample reads 3 captions). I plan to report both numbers side by side; please let me know if I misread the token accounting in `phase_real.csv`.
4. **Held-out scaling-law RMSE (`0.0093` vs `0.0109`):** `phase_law_predictions.csv` gives `0.0093` RMSE on the 85 `design == "random"` runs (matching the README) and `0.0109` across all 88 non-grid held-out runs when the 3 `320M`-sample `design == "long"` runs are included (matching `paper/tables.py`).
5. **`tab:captioner_scale` (`.838` COCO CIDEr):** Every README headline rebuilt from `paper/data/*.csv` except `.838` COCO CIDEr, which `docs/reproduce.md` marks as `typed`. Is there a per-seed CSV for that table that I missed?

Thank you for releasing such a clean and well-documented repository. If any of the points above miss context from your setup, I will gladly incorporate your response in the report.

Best regards,
Mohamed Ansar
