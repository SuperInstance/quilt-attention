// quilt-attention — training, receipts, verification.
//
// THE PATTERN (reused from SuperInstance/quilt-nn @1ae7977, code written fresh):
// every epoch of train() appends ONE entry to a sha256-linked chain,
//
//   { v, seq, epoch, loss, loss_sha, acc, acc_sha, weight_root_sha, prev, sha }
//
//   loss            the epoch's mean PRE-STEP loss measured along the pass (what the
//                   gradients were taken at) — a plain JS number for humans
//   loss_sha        sha256 over the portable scalar preimage of that number:
//                   "f64|8|<8-byte BE IEEE-754 hex>" (receipt format v2, quilt-attention#1)
//   acc             post-epoch token accuracy over the whole dataset (see accuracy())
//   acc_sha         sha256 over the same portable scalar preimage of acc
//   weight_root_sha sha256 over every weight's canonical bytes, in graph order
//   prev / sha      the chain links; sha = sha256 of the canonical entry JSON
//
// verify(chain) recomputes every link and every commitment. Honest limit, stated
// once: a bare hash chain cannot detect truncation of its TAIL without an externally
// anchored tip — same as any hash chain, including git. Everything else — edited
// losses, accuracies, weight roots, links, reordered or renumbered entries — is
// detected.
//
// THE TOY TASKS (vocab 8, len 6):
//   'copy'    y[i] = x[i]        — the head must learn (near-)identity attention
//   'reverse' y[i] = x[len-1-i]  — the head must learn the anti-diagonal permutation
// Without positional embeddings self-attention is permutation-equivariant and
// NEITHER task is representable; the P table is what buys position. That is a
// statement about the architecture, and the localization battery makes it visible:
// perturbing P moves every digest downstream of the posemb cells.
//
// ZERO dependencies, ESM, Node >= 18.

import {
  lcg, sha256hex, scalarSha, GENESIS,
  loadNet, initEnv, evaluateForward, evaluateBackward, applyTick, weightRootSha,
} from './attncells.mjs';

export { GENESIS };

// ── data makers (seeded; init and data from seeds, per the spec) ──────────────────
// makeTaskData({task, vocab, len, n, seed}) → [{x: Float64Array, y: Float64Array}]
// n seeded random sequences; targets per the task. The FIRST sequence is pinned to
// a ramp [0,1,...,len-1 mod vocab] so the training set always contains an ordered
// sequence (a shape SGD would otherwise have to stumble on).
export function makeTaskData({ task = 'copy', vocab = 8, len = 6, n = 16, seed = 7 } = {}) {
  if (task !== 'copy' && task !== 'reverse') throw new TypeError(`task must be 'copy' or 'reverse', got "${task}"`);
  if (!Number.isInteger(n) || n < 1) throw new TypeError('n must be a positive integer');
  const r = lcg(seed);
  const target = (x) => {
    const y = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = task === 'copy' ? x[i] : x[x.length - 1 - i];
    return y;
  };
  const data = [];
  const ramp = new Float64Array(len);
  for (let i = 0; i < len; i++) ramp[i] = i % vocab;
  data.push({ x: ramp, y: target(ramp) });
  for (let k = 1; k < n; k++) {
    const x = new Float64Array(len);
    for (let i = 0; i < len; i++) x[i] = Math.floor(r.next() * vocab);
    data.push({ x, y: target(x) });
  }
  return data;
}

// ── metrics ───────────────────────────────────────────────────────────────────────
// mean loss over the dataset with the CURRENT weights (forward only)
export function meanLoss(net, env, data) {
  let sum = 0;
  for (const s of data) {
    evaluateForward(net, env, s, undefined, false);
    sum += env[net.lossId][0];
  }
  return sum / data.length;
}

// post-epoch token accuracy: fraction of POSITIONS whose argmax(logits) equals the
// target token, averaged over samples (the honest per-token rate, not per-sequence).
export function accuracy(net, env, data) {
  let ok = 0;
  let total = 0;
  for (const s of data) {
    evaluateForward(net, env, s, undefined, false);
    const logits = env['logits'];
    const len = s.y.length;
    const V = logits.length / len;
    for (let i = 0; i < len; i++) {
      let best = 0;
      for (let j = 1; j < V; j++) if (logits[i * V + j] > logits[i * V + best]) best = j;
      if (best === s.y[i]) ok += 1;
      total += 1;
    }
  }
  return ok / total;
}

// full-batch mean analytic gradient over the dataset — the quantity a 'batch' tick
// applies, and the analytic side of the gradient check (T3).
export function fullBatchGrad(net, env, data) {
  const acc = new Map();
  for (const w of net.weightCells) acc.set(w.id, new Float64Array(w.params.shape[0] * w.params.shape[1]));
  for (const s of data) {
    evaluateForward(net, env, s, undefined, false);
    const g = evaluateBackward(net, env);
    for (const [id, arr] of g) {
      const dst = acc.get(id);
      if (dst && dst.length === arr.length) for (let i = 0; i < arr.length; i++) dst[i] += arr[i];
    }
  }
  for (const [, arr] of acc) for (let i = 0; i < arr.length; i++) arr[i] /= data.length;
  return acc;
}

// ── canonical receipts ────────────────────────────────────────────────────────────
function canonEntry(e) { // fixed key order — the canonical bytes of a receipt
  return JSON.stringify({
    v: e.v, seq: e.seq, epoch: e.epoch,
    loss: e.loss, loss_sha: e.loss_sha,
    acc: e.acc, acc_sha: e.acc_sha,
    weight_root_sha: e.weight_root_sha, prev: e.prev,
  });
}

// ── verification ─────────────────────────────────────────────────────────────────
export function verifyWhy(chain) {
  if (!Array.isArray(chain)) return { ok: false, at: 0, reason: 'chain is not an array' };
  if (chain.length === 0) return { ok: true, n: 0 }; // an empty chain trivially verifies
  let prev = GENESIS;
  for (let i = 0; i < chain.length; i++) {
    const e = chain[i];
    if (typeof e !== 'object' || e === null) return { ok: false, at: i, reason: 'entry is not an object' };
    if (e.v !== 2) return { ok: false, at: i, reason: `receipt format v${e.v} unsupported (v2 = portable scalar preimage "f64|8|<hex>"; v1 receipts predate the quilt-attention#1 preimage fix and verify only under the v0.1.0 code)` };
    if (e.seq !== i + 1) return { ok: false, at: i, reason: `seq is ${e.seq}, expected ${i + 1}` };
    if (!Number.isInteger(e.epoch) || e.epoch !== i) return { ok: false, at: i, reason: `epoch is ${e.epoch}, expected ${i} (epochs are contiguous from 0)` };
    if (typeof e.loss !== 'number' || !Number.isFinite(e.loss)) return { ok: false, at: i, reason: `loss ${e.loss} is not a finite number` };
    if (typeof e.acc !== 'number' || !Number.isFinite(e.acc) || e.acc < 0 || e.acc > 1) return { ok: false, at: i, reason: `acc ${e.acc} is not a number in [0,1]` };
    if (e.prev !== prev) return { ok: false, at: i, reason: `prev-link broken: ${e.prev} ≠ ${prev}` };
    if (e.loss_sha !== scalarSha(e.loss)) return { ok: false, at: i, reason: `loss_sha does not match the canonical bytes of loss ${e.loss}` };
    if (e.acc_sha !== scalarSha(e.acc)) return { ok: false, at: i, reason: `acc_sha does not match the canonical bytes of acc ${e.acc}` };
    if (!/^[0-9a-f]{64}$/.test(e.loss_sha) || !/^[0-9a-f]{64}$/.test(e.acc_sha) || !/^[0-9a-f]{64}$/.test(e.weight_root_sha)) {
      return { ok: false, at: i, reason: 'hash fields must be 64 lowercase hex chars' };
    }
    const expect = sha256hex(canonEntry(e));
    if (e.sha !== expect) return { ok: false, at: i, reason: 'entry sha mismatch (fields edited after sealing)' };
    prev = e.sha;
  }
  return { ok: true, n: chain.length };
}

export function verify(chain) {
  return verifyWhy(chain).ok;
}

// ── training ──────────────────────────────────────────────────────────────────────
// train(netOrGraph, {task, data, epochs, lr, seed, mode}) →
//   { metrics, chain, tip, env, finalLoss, finalAcc }
//
// mode 'sgd' (default): ONE tick per sample over a seeded Fisher–Yates shuffle of
//   the dataset. Stochasticity breaks the symmetry plateaus that stall full-batch
//   descent; the shuffle order comes from the SAME LCG as the init (the stream is
//   shared), so everything stays seed-only deterministic.
// mode 'batch': one tick per epoch on the mean gradient (fullBatchGrad).
//
// Deterministic end to end for a fixed seed: LCG init + shuffle, listed-order
// aggregation, IEEE-754 arithmetic, canonical hashing. Two runs, one byte stream
// (test T2).
export function train(netOrGraph, { data, epochs, lr, seed, mode = 'sgd' } = {}) {
  const net = netOrGraph.graph ? netOrGraph : loadNet(netOrGraph);
  if (!Array.isArray(data) || data.length === 0) throw new TypeError('train: data must be a non-empty array of {x, y}');
  if (!Number.isInteger(epochs) || epochs < 0) throw new TypeError('train: epochs must be a non-negative integer');
  if (!(typeof lr === 'number' && Number.isFinite(lr) && lr > 0)) throw new TypeError('train: lr must be a positive finite number');
  if (!Number.isInteger(seed)) throw new TypeError('train: seed must be an integer');
  if (mode !== 'sgd' && mode !== 'batch') throw new TypeError(`train: mode must be 'sgd' or 'batch', got "${mode}"`);

  const r = lcg(seed);
  const env = initEnv(net, seed, r); // continues the SAME LCG stream as the shuffle
  const chain = [];
  const metrics = [];
  let prev = GENESIS;

  const indices = data.map((_, i) => i);
  for (let epoch = 0; epoch < epochs; epoch++) {
    let loss;
    if (mode === 'batch') {
      loss = meanLoss(net, env, data); // pre-step mean loss (also binds env)
      const g = fullBatchGrad(net, env, data);
      applyTick(net, env, g, lr);
    } else {
      // Fisher–Yates over the sample order, driven by the training LCG
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(r.next() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      let total = 0;
      for (const idx of indices) {
        const s = data[idx];
        evaluateForward(net, env, s, undefined, false);
        total += env[net.lossId][0]; // pre-step: what this tick's gradient was taken at
        const g = evaluateBackward(net, env);
        applyTick(net, env, g, lr);
      }
      loss = total / data.length;
    }

    const entry = {
      v: 2,
      seq: epoch + 1,
      epoch,
      loss,
      loss_sha: scalarSha(loss),
      acc: accuracy(net, env, data),
      weight_root_sha: weightRootSha(net, env),
      prev,
    };
    entry.acc_sha = scalarSha(entry.acc);
    entry.sha = sha256hex(canonEntry(entry));
    chain.push(entry);
    metrics.push({ epoch, loss, acc: entry.acc });
    prev = entry.sha;
  }
  return {
    metrics,
    chain,
    tip: prev,
    env,
    finalLoss: chain.length ? chain[chain.length - 1].loss : undefined,
    finalAcc: chain.length ? chain[chain.length - 1].acc : undefined,
  };
}
