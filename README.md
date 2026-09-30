# quilt-attention

One-head self-attention, built as a **cell DAG** and trained for real — with zero
dependencies, hand-rolled backprop, a sha256 receipt chain per epoch, and faults
localized by cell digest. Node >= 18, ESM.

## Lineage

Conventions inherited from two fleet siblings (code written fresh, no shared source):

- **SuperInstance/cellgraph** — the cell convention: a graph of `{id, kind, inputs[],
  params}` nodes, each evaluated by a pure fn under the calling convention
  `f(env, *inputs, cell) -> Float64Array`; per-cell output digests; faults found by
  perturbing a cell and diffing witness digests downstream.
  **Documented deviation:** cellgraph seals digests with BLAKE2b; Node's `node:crypto`
  exposes no blake2b, so this repo uses **sha256 with the dtype tag INSIDE the preimage**
  (`"f64|<len>|<big-endian IEEE-754 bytes>",` see `arrayDigest`). Rule 2 of the
  cellgraph digest discipline is preserved — the dtype is hashed, not assumed.
- **SuperInstance/quilt-nn** — the receipt discipline: every epoch of training seals one
  entry of a sha256-linked chain, `weight_root_sha` commits all parameters, `prev/sha`
  link the chain, and `verify(chain)` recomputes every link. Same honest limit quilt-nn
  stated: a bare hash chain cannot detect truncation of its tail without an externally
  anchored tip; everything else (edited losses, forged accuracies, reweighted roots,
  broken links) is detected.

## The graph (`src/attncells.mjs`)

```
x,y      input cells (tokens / targets, len each)
E        weight — token embedding table (vocab × dim)
P        weight — positional embedding table (len × dim)
emb_i    embed   — row tokens[i] of E          (the lookup-table cells)
pos_i    posemb  — row i of P
X_i      add     — emb_i + pos_i
X        stack   — the len × dim sequence matrix
Wq,Wk,Wv weights — q/k/v projections
Q,K,V    matmul  — X·Wq, X·Wk, X·Wv
A        attn    — softmax(Q·Kᵀ/√dim)          (row-wise, max-subtracted)
H        matmul  — A·V                          (the weighted value sum)
Wo       weight  — output projection (dim × vocab)
logits   matmul  — H·Wo
loss     xent    — cross-entropy(logits, y), mean over positions
```

`P` carries a larger init scale than `E` (`posScale`, default 2.0) **by design**: without
positional signal self-attention is permutation-equivariant and neither task below is
representable — the positional part must be able to dominate the attention scores for
SGD to find the solution at all. Stated as an inductive bias, not hidden.

Backprop (`evaluateBackward`) is hand-rolled per cell kind and verified against central
finite differences (test T3, worst rel err < 1e-4 — actually ~1e-10).

## Training & receipts (`src/train.mjs`)

- Toy task: **sequence reversal** (`y[i] = x[len-1-i]`), vocab 6, len 4, 16 seeded
  sequences (first one pinned to a ramp), seed 7 everywhere.
- SGD, one tick per sample over a seeded Fisher–Yates shuffle; the shuffle shares the
  init LCG stream, so the whole run is seed-only deterministic.
- Per-epoch receipt: `{v, seq, epoch, loss, loss_sha, acc, acc_sha, weight_root_sha,
  prev, sha}` — `loss` is the mean pre-step loss (what the gradients were taken at),
  `acc` is the post-epoch per-token accuracy. `sha = sha256(canonical entry JSON)`.
- `verify(chain)` / `verifyWhy(chain)` recompute seq, epoch, links, every commitment.

## Fault localization (`src/localize.mjs`)

Perturb exactly one weight cell (+eps on every element), recompute every cell's
digest, and the **changed set must EQUAL the downstream topological slice** of that
cell — computed from the graph edges alone. Two documented honest caveats the battery
taught us:

1. **Granularity** — the witness can resolve FINER than the cell graph (posemb_i reads
   only row i of P), so the pin perturbs the whole cell to match cell granularity.
2. **Symmetry** — a uniform shift on ALL of `Wo` moves every logit of a position by the
   same constant, and max-subtracted cross-entropy is mathematically invariant to that.
   The loss digest can legitimately stay put on such directions (test T4 uses sample
   `data[1]`, where the per-logit rounding is visible). The witness records what a
   value IS, not what could flow.

## Run

```
npm test          # node --test test/att.test.mjs — six tests
node src/localize.mjs   # the localization battery, printed as a table
```

See `TEST-RECEIPT.md` for the actual recorded numbers of the v0.1.0 run.
