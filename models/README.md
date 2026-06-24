# Models Directory

This directory stores local trained model artifacts.

Expected artifact:

```text
models/ctr_model.pt
```

Generate it with:

```powershell
python -m src.train
```

For larger or production models, prefer Git LFS, release assets, or an external artifact store instead of committing binaries directly.
