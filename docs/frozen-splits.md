# Freeze acquisition-group splits

Create the split before extracting planes or patches. Each inventory row is one source field. `group_id` must be the highest verified dependency unit available from acquisition metadata—for example a well, specimen or acquisition batch. Do not invent this identity from filenames.

```bash
staining split examples/source_inventory_template.csv \
  --validation-groups 1 --test-groups 1 --seed 17 \
  --out-dir runs/frozen-split
```

The command requires explicit held-out group counts and at least one remaining training group. It assigns complete groups using a stable SHA-256 ranking, then writes `assignments.csv` and a fingerprinted `split.json`. Source row order does not affect the result. The output directory must not already exist, which prevents an accidental reroll or overwrite.

The fingerprint identifies the algorithm, seed, requested counts and exact source/group assignments. Keep this output with the experiment record and copy an approved small split file into version control only when the dataset terms permit it. Any change to sources, group identities, counts or seed creates a different fingerprint and therefore a different experiment.

This command enforces declared group boundaries; it cannot prove that the declared groups are biologically independent. Inspect the dataset provenance and acquisition metadata first. Test identities should remain unopened for model selection after the plan is frozen.
