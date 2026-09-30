# TEST-RECEIPT — quilt-attention v0.1.0

All numbers below are from an actual run of `node --test test/att.test.mjs`
(Node v24.21.0, linux x64) plus a standalone capture of the receipt chain.
Configuration: task **reverse**, vocab 6, len 4, dim 16, 16 samples, seed 7,
lr 0.1, SGD, 1200 epochs.

## T1 — training to the accuracy bar

| metric | value |
|---|---|
| final loss (epoch 1199) | 0.000037491951340619574 |
| final acc | 1.0 |
| acc ≥ 0.95 first at epoch | 8 (acc 1.0 by epoch 10) |
| epoch 0 receipt | loss 1.8584417829114086, acc 0.328125 |
| chain entries | 1200, `verify(chain)` = true |
| **chain tip** | `be700525d9554bdd60b4756f58dd1688d2dfd7771937ae49a48e517eb548617f` |
| wall time | ~3.3 s |

Tip receipt (epoch 1199): `loss_sha c9a1d01f…, acc_sha 98e370fc…,
weight_root_sha 02bca5c4…, prev 243138fc…`.

## T2 — determinism

Two full runs with the same seed produced **byte-identical chains**
(JSON.stringify equality, 200-epoch config) and equal tips.

## T3 — gradient check

Hand-rolled backprop vs central finite differences (h = 1e-5) on the 10
strongest-analytic-gradient parameters across E, P, Wq, Wk, Wv, Wo:
**worst rel err 9.847e-11** (at Wv[211]) — bar was < 1e-4.

## T4 — fault localization

`localizeBattery(trials=5, seed=7, eps=0.5)` on the seeded init, sample data[1]:

| weight | first moved | expected | changed | slice | verdict |
|---|---|---|---|---|---|
| P  | pos0   | pos0   | 17 | 17 | ok |
| Wo | logits | logits | 3  | 3  | ok |
| Wk | K      | K      | 6  | 6  | ok |
| Wo | logits | logits | 3  | 3  | ok |
| E  | emb0   | emb0   | 17 | 17 | ok |

**5/5 changed sets exactly equal the downstream topological slices**;
5/5 first-changed cell = direct consumer; negative control
(identical re-runs move nothing) = true.
VERDICT: "localization is exact — the changed-digest set IS the downstream slice".

## T5 — tamper detection

Forging entry 3's loss in a verified 10-entry chain → `verify()` = false,
`verifyWhy()` reason: entry sha mismatch (fields edited after sealing).

## NC1 — cell-order independence

Three seeded random topological orders of the same graph (all ≠ insertion
order): per-cell digests of all 29 cells **byte-identical** to the default
order's, including logits and loss.

## Suite

```
ℹ tests 6  ℹ pass 6  ℹ fail 0
```
