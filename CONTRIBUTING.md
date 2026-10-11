# Contributing

Contributions should keep scientific assumptions explicit and tests deterministic.

## Local setup

```bash
python -m pip install -e '.[test]'
pytest -q
mdstate demo --outdir demo_run
```

For changes involving the autoencoder, also install `.[latent]`. For HDBSCAN changes, install `.[density]`.

## Scientific contribution checklist

- Include unit and periodicity conventions for each new feature.
- Keep lagged pairs inside trajectory/segment boundaries.
- Add tests for invalid input and a synthetic known-answer case where practical.
- Do not use metadata, cluster labels, or post hoc outcomes as input features by default.
- State whether validation is frame-wise or trajectory-wise; for dynamical models prefer held-out trajectories/replicas.
- Do not describe a visual projection, silhouette score, or reconstruction loss alone as proof of a physical state or kinetic model.
- Include the exact command/configuration and random seed in examples.
- Do not add simulation trajectories, large raw data, environment installers, or unpublished manuscript figures to the package.

Licensing is unresolved in this alpha version; do not contribute third-party code unless its provenance and redistribution terms are clear.
