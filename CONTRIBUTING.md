# Contributing

## Development setup

```bash
pip install -r requirements.txt
pre-commit install     # installs black, flake8, mypy hooks
```

## Code style

- **Formatting**: `black --target-version py310 .`
- **Linting**: `flake8 --max-line-length=120 --extend-ignore=E203,W503`
- **Type checking**: `mypy --ignore-missing-imports --follow-imports=silent .`
- Run all three via: `pre-commit run --all-files`

## Testing

```bash
python -m test.test_clip         # CLIP embedding quality
python -m test.test_training     # 5-epoch pipeline sanity
```

## Pull request checklist

- [ ] `python -m test.test_clip` passes
- [ ] `python -m test.test_training` passes (5 epochs)
- [ ] `pre-commit run --all-files` passes
- [ ] New features include config defaults in `src/configs/config.yaml`
- [ ] Type hints added for all new function signatures
- [ ] `validate_config()` updated with any new required config keys
- [ ] No unused parameters in function signatures

## Project structure

| Directory | Purpose |
|---|---|
| `src/models/` | Generator, Discriminator, ZSL classifier |
| `src/training/` | GAN training loop (EMA, resume), WGAN-GP + feature matching losses |
| `src/evaluation/` | FID, ZSL classifier, GZSL calibrated stacking |
| `src/utils/` | Embeddings (CLIP/ensemble/GloVe), data loading, metrics tracker, visualization |
| `src/configs/` | YAML configuration (validated at startup) |
| `test/` | Sanity checks and test suites |
| `legacy/` | Original monolithic GloVe-based implementation |
