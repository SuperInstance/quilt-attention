// quilt-attention — one-head self-attention as a quilt cell DAG.
//
// LINEAGE
//   SuperInstance/cellgraph   the forward pass AS a cell graph: typed cells wired by
//                             named inputs, insertion order = topological order, a
//                             per-cell witness digest at every boundary, and fault
//                             localization by "which digests moved".
//   SuperInstance/quilt-nn    trained cell graphs: every epoch receipted into a
//                             sha256-linked chain, seeded determinism end to end.
//   quilt-attention           ATTENTION as cells: embed, q/k/v projections,
//                             softmax(QK^T/sqrt(d)), weighted value sum, output
//                             projection, cross-entropy — each a cell, each digested,
//                             so a fault in one weight is VISIBLE as the exact set of
//                             downstream digests that moved.
//
// ── THE HASH DEVIATION, STATED EXPLICITLY ────────────────────────────────────────
// cellgraph hashes tensors with BLAKE2b-256 over "dtype|shape|bytes". node:crypto
// (this repo is zero-dependency, Node >= 18) does NOT offer blake2b — verified:
// crypto.getHashes() contains no 'blake2b' on Node 24.21. The convention here is
// therefore sha256 over the SAME preimage discipline, named:
//
//     sha256-dtype-tagged
//
//   digest = sha256( `<dtype-tag>|<length>|<canonical f64 big-endian bytes>` )
//
// The dtype tag rides INSIDE the preimage exactly as cellgraph's dtype.str did, so an
// f32 array and an f64 array holding the same values can never produce the same
// witness. Everything downstream (localization, receipts) only ever compares digests
// produced by this one function, so the deviation is contained and honest.
//
// ── CELL CONVENTION (adopted from cellgraph) ─────────────────────────────────────
//   * a cell is f(env, *named_inputs) -> Float64Array
//   * weights live in env; a weight CELL's output is its own parameter array
//   * inputs[] name earlier cells: the array order IS a topological order
//   * every evaluation records a witness per cell: {cell, kind, shape, forecast,
//     digest, bytes} — the forecast (shape + L2 norm) is computed BEFORE the digest,
//     prediction preceding outcome, per the fleet's witness doctrine.
//
// ZERO dependencies, ESM, Node >= 18.

import { createHash } from 'node:crypto';

export const CELL_KINDS = ['input', 'weight', 'embed', 'posemb', 'add', 'stack', 'matmul', 'attn', 'xent'];
export const DTYPE_TAG = 'f64'; // every cell output in this repo is a Float64Array

export const GENESIS = '0'.repeat(64); // prev of the first receipt, per the fleet pattern

// ── canonical bytes & hashing ────────────────────────────────────────────────────
// f64hex: the 8 canonical big-endian IEEE-754 bytes of one number (-0 normalized to
// +0 so the bytes survive a JSON round-trip; the lesson quilt-nn recorded).
const _dv = new DataView(new ArrayBuffer(8));
const _HEX = Array.from({ length: 256 }, (_, i) => i.toString(16).padStart(2, '0'));

export function f64hex(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new TypeError(`f64hex needs a finite number, got ${x}`);
  if (Object.is(x, -0)) x = 0;
  _dv.setFloat64(0, x, false);
  let s = '';
  for (let i = 0; i < 8; i++) s += _HEX[_dv.getUint8(i)];
  return s;
}

export function sha256hex(str) {
  return createHash('sha256').update(str, 'utf8').digest('hex');
}

// THE cell-output digest: sha256-dtype-tagged (see the deviation note above).
export function arrayDigest(a) {
  if (!(a instanceof Float64Array)) throw new TypeError(`arrayDigest needs a Float64Array, got ${a?.constructor?.name}`);
  let bytes = '';
  for (let i = 0; i < a.length; i++) bytes += f64hex(a[i]);
  // preimage: dtype tag + length + bytes — dtype INSIDE, per cellgraph's rule 2
  return sha256hex(`${DTYPE_TAG}|${a.length}|${bytes}`);
}

// sha256 over the canonical bytes of one scalar (for receipt commitments)
export function scalarSha(x) {
  return sha256hex(Buffer.from(f64hex(x), 'hex').toString('latin1'));
}

// ── seeded PRNG ──────────────────────────────────────────────────────────────────
// LCG (Numerical Recipes), the family quilt-nn pinned: s = (1664525·s + 1013904223)
// mod 2^32. Portable arithmetic — same seed, same bytes, forever, no engine RNG.
export function lcg(seed) {
  let s = seed >>> 0;
  return {
    next() {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return s / 4294967296;
    },
    get state() { return s; },
  };
}

// ── the softmax pair ─────────────────────────────────────────────────────────────
// softmaxRows: row-wise softmax over an r×c row-major matrix, max-subtracted for
// stability. THE forward attention normalization.
export function softmaxRows(S, r, c) {
  const A = new Float64Array(r * c);
  for (let i = 0; i < r; i++) {
    const off = i * c;
    let m = -Infinity;
    for (let j = 0; j < c; j++) if (S[off + j] > m) m = S[off + j];
    let z = 0;
    for (let j = 0; j < c; j++) { const e = Math.exp(S[off + j] - m); A[off + j] = e; z += e; }
    for (let j = 0; j < c; j++) A[off + j] /= z;
  }
  return A;
}

// NEGATIVE CONTROL ONLY — never used by a valid graph. Deliberately broken
// normalization (divides by the GLOBAL sum instead of the row sum) so test NC2 can
// prove the row-sum check actually bites.
export function softmaxRowsBroken(S, r, c) {
  const A = new Float64Array(r * c);
  let g = 0;
  for (let i = 0; i < r * c; i++) { A[i] = Math.exp(S[i]); g += A[i]; }
  for (let i = 0; i < r * c; i++) A[i] /= g;
  return A;
}

// the NC2 pin: every row of an attention matrix sums to 1 within tol.
export function rowsSumToOne(A, r, c, tol = 1e-9) {
  for (let i = 0; i < r; i++) {
    let s = 0;
    for (let j = 0; j < c; j++) s += A[i * c + j];
    if (!(Math.abs(s - 1) <= tol)) return { ok: false, row: i, sum: s };
  }
  return { ok: true };
}

// ── cell functions ───────────────────────────────────────────────────────────────
// One pure function per kind, signature f(env, ...inputArrays, cell) -> Float64Array
// under the cellgraph calling convention f(env, *inputs) with the cell's own params
// bound by makeCellFn. Declared shapes live in params so the graph document is
// self-describing; every fn validates what it reads.

const FNS = {
  // input: bound per sample by evaluateForward; the fn re-reads the bound value.
  input: (env, cell) => {
    const v = env[cell.id];
    if (!(v instanceof Float64Array)) throw new Error(`input cell ${cell.id} has no bound value`);
    return v;
  },

  // weight: the parameter array IS the output; it persists in env across evaluations.
  weight: (env, cell) => {
    const w = env[cell.id];
    if (!(w instanceof Float64Array)) throw new Error(`weight cell ${cell.id} not initialized — call initEnv`);
    return w;
  },

  // embed: lookup-table read. inputs [tokens, table], params {pos, width}.
  // out = row tokens[pos] of the table, copied out.
  embed: (env, tokens, table, cell) => {
    const { pos, width } = cell.params;
    if (pos < 0 || pos >= tokens.length) throw new Error(`embed ${cell.id}: pos ${pos} outside tokens (len ${tokens.length})`);
    const tok = tokens[pos];
    if (!Number.isInteger(tok) || tok < 0) throw new Error(`embed ${cell.id}: token ${tok} at pos ${pos} is not a non-negative integer`);
    const vocab = table.length / width;
    if (!Number.isInteger(vocab)) throw new Error(`embed ${cell.id}: table length ${table.length} not a multiple of width ${width}`);
    if (tok >= vocab) throw new Error(`embed ${cell.id}: token ${tok} outside vocab ${vocab}`);
    return table.slice(tok * width, (tok + 1) * width);
  },

  // posemb: positional lookup. inputs [posTable], params {pos, width}.
  posemb: (env, posTable, cell) => {
    const { pos, width } = cell.params;
    const seqLen = posTable.length / width;
    if (!Number.isInteger(seqLen)) throw new Error(`posemb ${cell.id}: table length ${posTable.length} not a multiple of width ${width}`);
    if (pos < 0 || pos >= seqLen) throw new Error(`posemb ${cell.id}: pos ${pos} outside table (rows ${seqLen})`);
    return posTable.slice(pos * width, (pos + 1) * width);
  },

  // add: elementwise a + b (embedding + position).
  add: (env, a, b, cell) => {
    if (a.length !== b.length) throw new Error(`add ${cell.id}: length mismatch ${a.length} vs ${b.length}`);
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  },

  // stack: rows -> one (r×w) row-major matrix. inputs [row_0..row_{r-1}].
  // NOTE the calling convention f(env, *inputs, cell): the trailing cell object is
  // stripped here because rows is variadic.
  stack: (env, ...rest) => {
    const cell = rest[rest.length - 1];
    const rows = rest.slice(0, -1);
    if (rows.length === 0) throw new Error(`stack ${cell.id}: no inputs`);
    const w = rows[0].length;
    for (const r of rows) if (r.length !== w) throw new Error(`stack ${cell.id}: rows of unequal length`);
    const out = new Float64Array(rows.length * w);
    for (let i = 0; i < rows.length; i++) out.set(rows[i], i * w);
    return out;
  },

  // matmul: C = X·W, X is rows×p row-major, W is p×c row-major; params {rows}.
  matmul: (env, X, W, cell) => {
    const { rows } = cell.params;
    if (rows <= 0 || X.length % rows !== 0) throw new Error(`matmul ${cell.id}: X length ${X.length} not a multiple of rows ${rows}`);
    const p = X.length / rows;
    if (W.length % p !== 0) throw new Error(`matmul ${cell.id}: W length ${W.length} incompatible with inner dim ${p}`);
    const c = W.length / p;
    const out = new Float64Array(rows * c);
    for (let i = 0; i < rows; i++) {
      for (let k = 0; k < p; k++) {
        const xv = X[i * p + k];
        if (xv === 0) continue; // exact-zero skip: bit-identical result, fewer flops
        const woff = k * c;
        for (let j = 0; j < c; j++) out[i * c + j] += xv * W[woff + j];
      }
    }
    return out;
  },

  // attn: A = softmax_rows(Q·K^T / sqrt(dim)), params {len, dim}. No causal mask:
  // the copy/reversal tasks are encoder-style (every position may look everywhere).
  attn: (env, Q, K, cell) => {
    const { len, dim } = cell.params;
    if (Q.length !== len * dim || K.length !== len * dim) {
      throw new Error(`attn ${cell.id}: Q/K length ${Q.length}/${K.length} != len*dim ${len * dim}`);
    }
    const S = new Float64Array(len * len);
    const scale = 1 / Math.sqrt(dim);
    for (let i = 0; i < len; i++) {
      for (let j = 0; j < len; j++) {
        let s = 0;
        for (let k = 0; k < dim; k++) s += Q[i * dim + k] * K[j * dim + k];
        S[i * len + j] = s * scale;
      }
    }
    return softmaxRows(S, len, len);
  },

  // xent: cross-entropy. inputs [logits (len×V), target (len)]; out = [meanCE].
  // mean over positions — the gradient in backward is (p − onehot)/len accordingly.
  xent: (env, logits, target, cell) => {
    const len = target.length;
    if (logits.length % len !== 0) throw new Error(`xent ${cell.id}: logits length ${logits.length} incompatible with ${len} targets`);
    const V = logits.length / len;
    const out = new Float64Array(1);
    let ce = 0;
    for (let i = 0; i < len; i++) {
      const t = target[i];
      if (!Number.isInteger(t) || t < 0 || t >= V) throw new Error(`xent ${cell.id}: target ${t} at pos ${i} outside vocab ${V}`);
      const off = i * V;
      let m = -Infinity;
      for (let j = 0; j < V; j++) if (logits[off + j] > m) m = logits[off + j];
      let z = 0;
      for (let j = 0; j < V; j++) z += Math.exp(logits[off + j] - m);
      ce += -(logits[off + t] - m) + Math.log(z);
    }
    out[0] = ce / len;
    return out;
  },
};

// bind params: the cellgraph calling convention f(env, *inputs) with the cell's own
// params riding on the closure.
export function makeCellFn(cell) {
  const base = FNS[cell.kind];
  if (!base) throw new Error(`unknown cell kind "${cell.kind}"`);
  return (env, ...ins) => base(env, ...ins, cell);
}

// ── validation ───────────────────────────────────────────────────────────────────
// A graph must be a DAG whose cells reference only EARLIER cells: the array order is
// itself a topological order (the cellgraph insight, also quilt-nn's rule).
export function validateGraph(graph) {
  if (!graph || !Array.isArray(graph.cells)) throw new TypeError('graph must be {cells: [...]}');
  const seen = new Set();
  for (const c of graph.cells) {
    if (!c || typeof c.id !== 'string' || !c.id) throw new TypeError(`cell without id: ${JSON.stringify(c)}`);
    if (seen.has(c.id)) throw new Error(`duplicate cell id ${c.id}`);
    if (!CELL_KINDS.includes(c.kind)) throw new Error(`cell ${c.id}: unknown kind "${c.kind}"`);
    if (!Array.isArray(c.inputs)) throw new Error(`cell ${c.id}: inputs must be an array`);
    for (const dep of c.inputs) {
      if (!seen.has(dep)) throw new Error(`cell ${c.id} inputs ${dep}: not defined earlier (graph is not in topological order)`);
    }
    seen.add(c.id);
  }
  for (const c of graph.cells) {
    const p = c.params || {};
    if (c.kind === 'weight' && !(Array.isArray(p.shape) && p.shape.length === 2 && p.shape.every(Number.isInteger) && p.shape[0] > 0 && p.shape[1] > 0 && typeof p.scale === 'number' && p.scale > 0)) {
      throw new Error(`weight cell ${c.id}: params need shape [r,c] of positive integers and a positive scale`);
    }
    if ((c.kind === 'embed' || c.kind === 'posemb') && !(Number.isInteger(p.pos) && Number.isInteger(p.width) && p.width > 0)) {
      throw new Error(`${c.kind} cell ${c.id}: params need integer pos and positive integer width`);
    }
    if (c.kind === 'matmul' && !(Number.isInteger(p.rows) && p.rows > 0)) {
      throw new Error(`matmul cell ${c.id}: params.rows must be a positive integer`);
    }
    if (c.kind === 'attn' && !(Number.isInteger(p.len) && Number.isInteger(p.dim) && p.len > 0 && p.dim > 0)) {
      throw new Error(`attn cell ${c.id}: params need positive integer len and dim`);
    }
  }
  return graph;
}

// ── topological orders ───────────────────────────────────────────────────────────
// 'insertion'  the written order (always valid, always used in training)
// 'kahn-fifo' / 'kahn-lifo'  Kahn's algorithm, queue / stack — genuinely different,
//              equally valid orders.
// randomTopoOrder: Fisher–Yates over Kahn's ready-set with a seeded LCG — literally a
// SHUFFLE of valid insertion orders, for the order-independence control (NC1).
// Evaluation semantics never depend on the order: every cell reads only its own
// listed inputs. That is what makes NC1 a property of the design, not an accident.
export function topoOrder(graph, variant = 'insertion') {
  validateGraph(graph);
  const cells = graph.cells;
  if (variant === 'insertion') return cells.map((c) => c.id);
  if (variant !== 'kahn-fifo' && variant !== 'kahn-lifo') throw new Error(`unknown topo variant "${variant}"`);
  const adj = new Map(cells.map((c) => [c.id, []]));
  const indeg = new Map(cells.map((c) => [c.id, c.inputs.length]));
  for (const c of cells) for (const dep of c.inputs) adj.get(dep).push(c.id);
  const ready = cells.filter((c) => indeg.get(c.id) === 0).map((c) => c.id);
  const out = [];
  while (ready.length) {
    const id = variant === 'kahn-lifo' ? ready.pop() : ready.shift();
    out.push(id);
    for (const d of adj.get(id)) {
      indeg.set(d, indeg.get(d) - 1);
      if (indeg.get(d) === 0) ready.push(d);
    }
  }
  if (out.length !== cells.length) throw new Error('graph contains a cycle');
  return out;
}

export function randomTopoOrder(graph, rng) {
  validateGraph(graph);
  const cells = graph.cells;
  const adj = new Map(cells.map((c) => [c.id, []]));
  const indeg = new Map(cells.map((c) => [c.id, c.inputs.length]));
  for (const c of cells) for (const dep of c.inputs) adj.get(dep).push(c.id);
  const ready = cells.filter((c) => indeg.get(c.id) === 0).map((c) => c.id);
  const out = [];
  while (ready.length) {
    const pick = Math.floor(rng.next() * ready.length);
    const [id] = ready.splice(pick, 1);
    out.push(id);
    for (const d of adj.get(id)) {
      indeg.set(d, indeg.get(d) - 1);
      if (indeg.get(d) === 0) ready.push(d);
    }
  }
  if (out.length !== cells.length) throw new Error('graph contains a cycle');
  return out;
}

// ── the net: a loaded graph ───────────────────────────────────────────────────────
export function loadNet(graph) {
  validateGraph(graph);
  const byId = new Map(graph.cells.map((c) => [c.id, c]));
  const net = {
    graph,
    byId,
    order: graph.cells.map((c) => c.id),
    inputCells: graph.cells.filter((c) => c.kind === 'input'),
    weightCells: graph.cells.filter((c) => c.kind === 'weight'),
    lossCell: graph.cells.find((c) => c.kind === 'xent') || null,
    attnCells: graph.cells.filter((c) => c.kind === 'attn'),
    cellFns: new Map(),
  };
  for (const c of graph.cells) net.cellFns.set(c.id, makeCellFn(c));
  if (!net.lossCell) throw new Error('net needs exactly one xent loss cell');
  for (const c of net.inputCells) {
    if (!['x', 'y'].includes(c.params?.slot)) throw new Error(`input cell ${c.id}: params.slot must be "x" or "y"`);
  }
  net.lossId = net.lossCell.id;
  return net;
}

// ── seeded initialization ────────────────────────────────────────────────────────
// One LCG, weights consumed in graph order: value = (2u − 1) · scale (quilt-nn's
// convention). Same seed → same initial weight bytes.
export function initEnv(net, seed, stream) {
  const r = stream || lcg(seed);
  const env = {};
  for (const c of net.graph.cells) env[c.id] = null;
  for (const w of net.weightCells) {
    const n = w.params.shape[0] * w.params.shape[1];
    const arr = new Float64Array(n);
    for (let i = 0; i < n; i++) arr[i] = (2 * r.next() - 1) * w.params.scale;
    env[w.id] = arr;
  }
  return env;
}

// ── forward evaluation + witness ─────────────────────────────────────────────────
// env maps cellId -> Float64Array. Weights persist; activations are recomputed per
// sample. `order` defaults to insertion; ANY valid topo order gives byte-identical
// outputs (NC1). With collectWitness (default true) the call returns the witness:
// one {cell, kind, shape, forecast, digest, bytes} per cell, in evaluation order —
// the forecast (shape + L2 norm) is computed BEFORE the digest, prediction
// preceding outcome. Training passes collectWitness=false: the witness is a
// measurement instrument, and hashing every activation of every sample of every
// epoch would spend the whole budget on hex strings instead of arithmetic. The
// witness is always collected wherever a digest is MEANT to be read (localization,
// NC1, examples); the receipts commit weights directly via weightRootSha.
export function evaluateForward(net, env, sample, order, collectWitness = true) {
  const seq = order || net.order;
  // bind sample inputs wherever they sit in the order
  for (const c of net.inputCells) {
    const v = c.params.slot === 'x' ? sample.x : sample.y;
    if (!(v instanceof Float64Array)) throw new TypeError(`sample.${c.params.slot} must be a Float64Array`);
    env[c.id] = v;
  }
  const witness = collectWitness ? [] : null;
  for (const id of seq) {
    const c = net.byId.get(id);
    const args = c.inputs.map((d) => env[d]);
    for (let k = 0; k < args.length; k++) {
      if (!(args[k] instanceof Float64Array)) throw new Error(`cell ${id}: input ${c.inputs[k]} has no value (order not topological?)`);
    }
    const out = net.cellFns.get(id)(env, ...args);
    if (!(out instanceof Float64Array)) throw new Error(`cell ${id} did not produce a Float64Array`);
    for (let k = 0; k < out.length; k++) {
      if (!Number.isFinite(out[k])) throw new Error(`cell ${id} produced a non-finite value at [${k}] (divergence or bad data)`);
    }
    env[id] = out;
    if (collectWitness) {
      // forecast BEFORE the digest — the witness predicts, then commits
      let norm = 0;
      for (let k = 0; k < out.length; k++) norm += out[k] * out[k];
      const forecast = { shape: [out.length], norm: Math.sqrt(norm) };
      witness.push({ cell: id, kind: c.kind, shape: [out.length], forecast, digest: arrayDigest(out), bytes: out.length * 8 });
    }
  }
  return witness;
}

// convenience: run forward, return {witness, out}
export function forward(net, env, sample, order) {
  const witness = evaluateForward(net, env, sample, order);
  return { witness, out: env[net.lossId] };
}

// ── backward: analytic gradients, by hand, through the SAME graph ─────────────────
// Reverse-mode over the cells in REVERSE topo order. grad maps cellId ->
// dLoss/d(cellOutput) as Float64Array; weight gradients ACCUMULATE across consumers
// (the embedding table is read by `len` embed cells; X is read by q/k/v).
// No autograd tape, no closure zoo: the local derivative of each kind is a property
// of the kind, exactly as in quilt-nn — generalized here from scalars to arrays.
export function evaluateBackward(net, env, order) {
  const seq = order || net.order;
  const grad = new Map();
  const ensure = (id, n) => {
    if (!grad.has(id)) grad.set(id, new Float64Array(n));
    return grad.get(id);
  };
  const bump = (id, g) => { // ACCUMULATE g into grad[id]
    const dst = ensure(id, g.length);
    if (dst.length !== g.length) throw new Error(`gradient shape clash at ${id}: ${dst.length} vs ${g.length}`);
    for (let i = 0; i < g.length; i++) dst[i] += g[i];
  };
  grad.set(net.lossId, new Float64Array([1]));
  for (let i = seq.length - 1; i >= 0; i--) {
    const id = seq[i];
    const g = grad.get(id);
    if (!g) continue;
    const c = net.byId.get(id);
    switch (c.kind) {
      case 'xent': {
        // loss = mean_i CE(softmax(logits_i), y_i); dlogits = (p − onehot)/len
        const [logitsId, targetId] = c.inputs;
        const logits = env[logitsId];
        const target = env[targetId];
        const len = target.length;
        const V = logits.length / len;
        const dLogits = new Float64Array(logits.length);
        for (let p = 0; p < len; p++) {
          const off = p * V;
          let m = -Infinity;
          for (let j = 0; j < V; j++) if (logits[off + j] > m) m = logits[off + j];
          let z = 0;
          for (let j = 0; j < V; j++) z += Math.exp(logits[off + j] - m);
          for (let j = 0; j < V; j++) dLogits[off + j] = Math.exp(logits[off + j] - m) / z / len;
          dLogits[off + target[p]] -= 1 / len;
        }
        bump(logitsId, dLogits); // targets are data: no gradient flows to them
        break;
      }
      case 'matmul': {
        // C = X·W (X: rows×p, W: p×c): dX = dC·W^T, dW = X^T·dC
        const [xId, wId] = c.inputs;
        const X = env[xId];
        const W = env[wId];
        const rows = c.params.rows;
        const p = X.length / rows;
        const cc = W.length / p;
        const dC = g;
        const dX = new Float64Array(X.length);
        const dW = new Float64Array(W.length);
        for (let a = 0; a < rows; a++) {
          for (let k = 0; k < p; k++) {
            let s = 0;
            for (let j = 0; j < cc; j++) { s += dC[a * cc + j] * W[k * cc + j]; dW[k * cc + j] += X[a * p + k] * dC[a * cc + j]; }
            dX[a * p + k] = s;
          }
        }
        bump(xId, dX);
        bump(wId, dW);
        break;
      }
      case 'attn': {
        // A = softmax_rows(S), S = Q·K^T/√dim: dS_i = A_i ⊙ (dA_i − (A_i·dA_i));
        // dQ = dS·K/√dim, dK = dS^T·Q/√dim
        const [qId, kId] = c.inputs;
        const Q = env[qId];
        const K = env[kId];
        const { len, dim } = c.params;
        const A = env[id];
        const dS = new Float64Array(len * len);
        for (let a = 0; a < len; a++) {
          let dot = 0;
          for (let j = 0; j < len; j++) dot += A[a * len + j] * g[a * len + j];
          for (let j = 0; j < len; j++) dS[a * len + j] = A[a * len + j] * (g[a * len + j] - dot);
        }
        const scale = 1 / Math.sqrt(dim);
        const dQ = new Float64Array(Q.length);
        const dK = new Float64Array(K.length);
        for (let a = 0; a < len; a++) {
          for (let j = 0; j < len; j++) {
            const s = dS[a * len + j] * scale;
            if (s === 0) continue;
            for (let k = 0; k < dim; k++) {
              dQ[a * dim + k] += s * K[j * dim + k];
              dK[j * dim + k] += s * Q[a * dim + k];
            }
          }
        }
        bump(qId, dQ);
        bump(kId, dK);
        break;
      }
      case 'add': {
        const [aId, bId] = c.inputs;
        bump(aId, g);
        bump(bId, g);
        break;
      }
      case 'stack': {
        const w = g.length / c.inputs.length;
        for (let k = 0; k < c.inputs.length; k++) bump(c.inputs[k], g.subarray(k * w, (k + 1) * w));
        break;
      }
      case 'embed': {
        // out = row tokens[pos] of table: dtable row += dOut; tokens are data.
        const [tokensId, tableId] = c.inputs;
        const tokens = env[tokensId];
        const tok = tokens[c.params.pos];
        const { width } = c.params;
        const dT = ensure(tableId, env[tableId].length);
        for (let k = 0; k < width; k++) dT[tok * width + k] += g[k];
        break;
      }
      case 'posemb': {
        const [tableId] = c.inputs;
        const { pos, width } = c.params;
        const dT = ensure(tableId, env[tableId].length);
        for (let k = 0; k < width; k++) dT[pos * width + k] += g[k];
        break;
      }
      case 'input':
      case 'weight':
        // terminals: weight gradients were accumulated by their consumers above
        break;
      default:
        throw new Error(`backward cannot handle kind "${c.kind}" (${id})`);
    }
  }
  return grad;
}

// gradient of a weight cell from a backward pass (zeros if the weight is untouched
// on this sample — e.g. an embedding row whose token never appeared)
export function weightGrad(gradMap, net, weightId) {
  const w = net.byId.get(weightId);
  const n = w.params.shape[0] * w.params.shape[1];
  return gradMap.get(weightId) || new Float64Array(n);
}

// one SGD tick: weight -= lr * grad for every weight cell (in graph order).
// Returns Σ|Δw| — the diagnostic quilt-nn's tick cell keeps.
export function applyTick(net, env, gradMap, lr) {
  if (!(typeof lr === 'number' && Number.isFinite(lr) && lr > 0)) throw new TypeError(`lr must be a positive finite number, got ${lr}`);
  let total = 0;
  for (const w of net.weightCells) {
    const g = gradMap.get(w.id);
    if (!g) continue;
    const arr = env[w.id];
    for (let i = 0; i < arr.length; i++) {
      const d = lr * g[i];
      arr[i] -= d;
      total += Math.abs(d);
    }
  }
  return total;
}

// ── the weight root: every weight's canonical bytes, in graph order ───────────────
export function weightRootSha(net, env) {
  let buf = '';
  for (const w of net.weightCells) {
    const arr = env[w.id];
    buf += `${w.id}\n`;
    for (let i = 0; i < arr.length; i++) buf += `${f64hex(arr[i])}\n`;
  }
  return sha256hex(buf);
}

// ── builders ─────────────────────────────────────────────────────────────────────
// buildAttention({vocab, len, dim, posScale}) — one-head encoder self-attention as a
// cell DAG:
//
//   x,y            input cells (tokens / targets, len each)
//   E              weight cell: the token embedding table (vocab × dim)
//   P              weight cell: the positional embedding table (len × dim)
//   emb_i          embed: row tokens[i] of E         (the "lookup table cells")
//   pos_i          posemb: row i of P
//   X_i            add: emb_i + pos_i
//   X              stack: the len × dim sequence matrix
//   Wq,Wk,Wv       weight cells — the q/k/v projections' parameters
//   Q,K,V          matmul: X·Wq, X·Wk, X·Wv
//   A              attn: softmax(Q·K^T/√dim)
//   H              matmul: A·V (the weighted value sum)
//   Wo             weight cell: output projection (dim × vocab)
//   logits         matmul: H·Wo
//   loss           xent(logits, y)
//
// Xavier-uniform scale per weight: sqrt(6/(fanIn+fanOut)).
// P carries a larger scale than E BY DESIGN (posScale, default 2.0): the positional
// part must be able to dominate the attention scores for copy/reversal to be
// findable by SGD at all. This is an inductive bias, stated here and in the README.
export function buildAttention({ vocab = 8, len = 6, dim = 16, posScale = 2.0 } = {}) {
  if (!Number.isInteger(vocab) || vocab < 2) throw new TypeError('vocab must be an integer >= 2');
  if (!Number.isInteger(len) || len < 2) throw new TypeError('len must be an integer >= 2');
  if (!Number.isInteger(dim) || dim < 1) throw new TypeError('dim must be a positive integer');
  if (!(typeof posScale === 'number' && posScale > 0)) throw new TypeError('posScale must be a positive number');
  const cells = [];
  const W = (id, rows, cols, scale) => cells.push({ id, kind: 'weight', inputs: [], params: { shape: [rows, cols], scale } });

  cells.push({ id: 'x', kind: 'input', inputs: [], params: { slot: 'x' } });
  cells.push({ id: 'y', kind: 'input', inputs: [], params: { slot: 'y' } });
  W('E', vocab, dim, Math.sqrt(6 / (vocab + dim)));
  W('P', len, dim, posScale);

  const rowIds = [];
  for (let i = 0; i < len; i++) {
    cells.push({ id: `emb${i}`, kind: 'embed', inputs: ['x', 'E'], params: { pos: i, width: dim } });
    cells.push({ id: `pos${i}`, kind: 'posemb', inputs: ['P'], params: { pos: i, width: dim } });
    cells.push({ id: `X${i}`, kind: 'add', inputs: [`emb${i}`, `pos${i}`], params: {} });
    rowIds.push(`X${i}`);
  }
  cells.push({ id: 'X', kind: 'stack', inputs: rowIds, params: {} });

  W('Wq', dim, dim, Math.sqrt(6 / (2 * dim)));
  W('Wk', dim, dim, Math.sqrt(6 / (2 * dim)));
  W('Wv', dim, dim, Math.sqrt(6 / (2 * dim)));
  cells.push({ id: 'Q', kind: 'matmul', inputs: ['X', 'Wq'], params: { rows: len } });
  cells.push({ id: 'K', kind: 'matmul', inputs: ['X', 'Wk'], params: { rows: len } });
  cells.push({ id: 'V', kind: 'matmul', inputs: ['X', 'Wv'], params: { rows: len } });
  cells.push({ id: 'A', kind: 'attn', inputs: ['Q', 'K'], params: { len, dim } });
  cells.push({ id: 'H', kind: 'matmul', inputs: ['A', 'V'], params: { rows: len } });
  W('Wo', dim, vocab, Math.sqrt(6 / (dim + vocab)));
  cells.push({ id: 'logits', kind: 'matmul', inputs: ['H', 'Wo'], params: { rows: len } });
  cells.push({ id: 'loss', kind: 'xent', inputs: ['logits', 'y'], params: {} });
  return { cells };
}
