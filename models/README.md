# Models Directory

This directory stores local trained model artifacts.

Expected artifacts:

```text
models/ctr_model.pt
models/training_report.json
```

Generate them with:

```powershell
python -m src.train
```

`ctr_model.pt` is used by CTR reranking. `training_report.json` stores dataset split, training config, loss, and validation AUC for the demo UI.

For larger or production artifacts, prefer Git LFS, release assets, or an external artifact store instead of committing binaries directly.
