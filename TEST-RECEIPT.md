# TEST-RECEIPT — quilt-attention v0.1.1

All numbers below are from an actual run of `npm test`
(Node v24.21.0, linux x64) plus a standalone capture of the receipt chain.
Configuration: task **reverse**, vocab 6, len 4, dim 16, 16 samples, seed 7,
lr 0.1, SGD, 1200 epochs.

**v0.1.1 note (quilt-attention#1):** `scalarSha`'s preimage changed from hashing
a latin1-re-encoded byte buffer to the legible tagged string
`"f64|8|<big-endian IEEE-754 hex>"` — the form `arrayDigest` always used. Every
epoch receipt is now format **v2**; `verify()` rejects v1 with a named reason.
Training arithmetic is untouched — every loss/accuracy number below is
**identical to the v0.1.0 receipt**; only sha commitments changed (and
`weight_root_sha`, which never used the broken path, is byte-identical).

## C0 — cross-runtime scalar conformance (new in v0.1.1)

`test/conformance.test.mjs` (node) and `test/conformance.py` (python3, stdlib
only) hash the shared 32-scalar adversarial fixture
(`test/scalar-fixture.json`: 0, −0, 1, 1.0, −1, π, e, 1/3, 0.1, 1e-7, 1e21,
2^53, 2^53+1-as-float, ±min-denormal, min-normal, max-double, the issue's own
loss scalar, …) and assert **32/32 digests identical** to the committed
agreement list (`test/scalar-expected-digests.txt`), plus 3/3 non-finite
sentinels rejected fail-closed in both runtimes and the −0 ≡ 0 normalization.
The v0.1.0 code disagreed 5/5 on the same pairs (red receipt in the issue).

## T1 — training to the accuracy bar

| metric | value |
|---|---|
| final loss (epoch 1199) | 0.000037491951340619574 |
| final acc | 1.0 |
| acc ≥ 0.95 first at epoch | 8 (acc 1.0 by epoch 10) |
| epoch 0 receipt | loss 1.8584417829114086, acc 0.328125 |
| chain entries | 1200, `verify(chain)` = true |
| **chain tip** | `0590dff6fd6623cde7a042f5b1f9aade5ecb0779ca1720697281b051659a255a` |
| wall time | ~4.8 s (suite incl. conformance) |

Tip receipt (epoch 1199): `loss_sha 46a8ef6b…, acc_sha 5283972f…,
weight_root_sha 02bca5c4…, prev f15b9a18…`.

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
Also now: downgrading an entry's `v` to 1 → `verify()` = false, reason names the
receipt format.

## NC1 — cell-order independence

Three seeded random topological orders of the same graph (all ≠ insertion
order): per-cell digests of all 29 cells **byte-identical** to the default
order's, including logits and loss.

## Suite

```
npm test   (node --test test/*.test.mjs)
ℹ tests 10  ℹ pass 10  ℹ fail 0
```

plus `python3 test/conformance.py`: 32/32 + 3/3, exit 0.
