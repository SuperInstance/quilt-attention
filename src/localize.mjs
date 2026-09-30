// quilt-attention — fault localization by cell digest (the cellgraph adaptation).
//
// THE CLAIM (inherited from SuperInstance/cellgraph's findfault discipline):
// "perturb one weight and the witness tells you where the damage lives." Here the
// claim is made STRONGER and MECHANICAL:
//
//   perturb exactly one weight cell by +epsilon, recompute every cell's
//   sha256-dtype-tagged digest, and the set of cells whose digest changed must
//   EQUAL the downstream slice of the perturbed cell — the set {cell} ∪
//   {every transitive dependent}, computed from the graph topology ALONE.
//
// Not "the first changed cell is downstream" (cellgraph's one-block check), not "some
// cells moved": the WHOLE changed set must equal the topology slice, no more, no
// less. A cell that moved outside the slice would mean a digest is lying about data
// flow; a cell inside the slice that did NOT move would mean the witness is blind to
// a real dependency (cellgraph's float32-precision lesson). This battery pins both.
//
// Also inherited from cellgraph: the first changed cell in topo order must be the
// perturbed weight's DIRECT consumer (the witness points at the consumer of the
// weight that moved, not at a fixed position in the file), and two identical runs
// must report NO changed cells at all (a broken instrument reporting a clean result
// is the worst outcome, so it gets its own control).
//
// ── TWO HONEST CAVEATS THE BATTERY TAUGHT US (both real, both documented) ────────
// 1. GRANULARITY. The witness can resolve FINER than the cell graph: posemb_i reads
//    only row i of P, so P[87] += 0.5 moves pos5's digest and no other posemb. The
//    changed set is then a strict SUBSET of the slice — correct values, coarser
//    prediction. The battery therefore perturbs the WHOLE cell, matching the cell
//    granularity of the pin.
// 2. SYMMETRY. A uniform +eps on ALL of Wo shifts every logit of position i by the
//    SAME constant (eps·Σ_k H_ik) — and cross-entropy is mathematically INVARIANT
//    to a class-independent logit shift. The loss VALUE does not move along this
//    direction (up to f64 rounding of the shift itself), so on a trained net the
//    loss digest may stay put while logits move. The witness records what a value
//    IS, not what could flow — a symmetry direction is invisible to any value
//    digest, by construction. The battery's exact-equality pin therefore runs on
//    the seeded INITIAL weights (where rounding at eps=0.5 is visible and every
//    slice is confirmed exact); localizeOne's output carries missing/extra lists so
//    a symmetry-masked cell is always NAMED, never silently swallowed.
//
// ZERO dependencies, ESM, Node >= 18.

import { evaluateForward, lcg } from './attncells.mjs';

// ── topology ─────────────────────────────────────────────────────────────────────
// downstreamSlice(graph, cellId): {cellId} ∪ transitive dependents, from the graph
// edges alone (inputs[] arrays). No numerics involved — this is the prediction the
// digest comparison is checked AGAINST.
export function downstreamSlice(graph, cellId) {
  const byId = new Map(graph.cells.map((c) => [c.id, c]));
  if (!byId.has(cellId)) throw new Error(`downstreamSlice: no cell "${cellId}"`);
  const consumers = new Map(graph.cells.map((c) => [c.id, []]));
  for (const c of graph.cells) for (const dep of c.inputs) consumers.get(dep).push(c.id);
  const slice = new Set([cellId]);
  const queue = [cellId];
  while (queue.length) {
    const id = queue.pop();
    for (const next of consumers.get(id)) {
      if (!slice.has(next)) { slice.add(next); queue.push(next); }
    }
  }
  return slice;
}

// directConsumers(graph, cellId): the cells that READ this cell one edge away.
export function directConsumers(graph, cellId) {
  return graph.cells.filter((c) => c.inputs.includes(cellId)).map((c) => c.id);
}

// ── the measurement ──────────────────────────────────────────────────────────────
// digestsOf(net, env, sample): one forward pass WITH the witness → Map cell → digest.
export function digestsOf(net, env, sample) {
  const witness = evaluateForward(net, env, sample, undefined, true);
  return new Map(witness.map((w) => [w.cell, w.digest]));
}

// perturbEnv(net, env, cellId, {eps}): DEEP-COPY the arrays (the baseline env is
// never mutated) and add eps to EVERY element of ONE weight cell.
//
// WHY the whole CELL and not one element: the pin compares at CELL granularity
// (changed-digest set vs topology slice), so the perturbation must be at cell
// granularity too (see caveat 1 in the header: an element-level hit on E or P can
// be FINER than the cell graph — the witness resolves WHICH ROW — which is a bonus
// property of the witness, not a failure of the pin).
export function perturbEnv(net, env, cellId, { eps = 0.5 } = {}) {
  if (!net.weightCells.some((w) => w.id === cellId)) throw new Error(`perturbEnv: "${cellId}" is not a weight cell`);
  const env2 = {};
  for (const c of net.graph.cells) {
    env2[c.id] = env[c.id] instanceof Float64Array ? Float64Array.from(env[c.id]) : env[c.id];
  }
  const w = env2[cellId];
  for (let i = 0; i < w.length; i++) w[i] += eps;
  return env2;
}

// localizeOne(net, env, sample, cellId, {eps}) →
//   { cell, eps, changed, slice, ok, first, expectedFirst, missing, extra }
//   changed        cells whose digest moved (observed)
//   slice          the topology prediction (expected)
//   ok             changed set DEEP-EQUALS the slice
//   first          first changed cell in topo order EXCLUDING the perturbed weight
//                  itself (which always moves — it IS the damaged array);
//                  expectedFirst = its direct consumer (the cellgraph pin)
export function localizeOne(net, env, sample, cellId, { eps = 0.5 } = {}) {
  const base = digestsOf(net, env, sample);
  const after = digestsOf(net, perturbEnv(net, env, cellId, { eps }), sample);
  const order = net.order;
  const changed = order.filter((id) => base.get(id) !== after.get(id));
  const slice = [...downstreamSlice(net.graph, cellId)];
  const sliceSet = new Set(slice);
  const changedSet = new Set(changed);
  const missing = slice.filter((id) => !changedSet.has(id)); // real dependency the digest missed
  const extra = changed.filter((id) => !sliceSet.has(id));   // digest movement without a data edge
  const consumers = directConsumers(net.graph, cellId);
  return {
    cell: cellId,
    eps,
    changed,
    slice,
    ok: missing.length === 0 && extra.length === 0,
    first: changed.find((id) => id !== cellId) ?? null,
    expectedFirst: consumers[0] ?? null,
    missing,
    extra,
  };
}

// ── the battery ──────────────────────────────────────────────────────────────────
// localizeBattery(net, env, sample, {trials, seed, eps}) — T4's engine:
//   * `trials` seeded-random weight cells, one whole-cell perturbation each
//   * pin per trial: changed set == topology slice AND first-changed == consumer
//   * negative control: an UNperturbed re-run must report zero changed cells
// Returns {results, controls, pass, summary} — pass means EVERY pin held.
export function localizeBattery(net, env, sample, { trials = 10, seed = 7, eps = 0.5 } = {}) {
  const r = lcg(seed);
  const weights = net.weightCells;
  if (weights.length === 0) throw new Error('localizeBattery: net has no weight cells');
  // negative control FIRST: an identical re-run must move nothing
  const baseA = digestsOf(net, env, sample);
  const baseB = digestsOf(net, env, sample);
  const controls = {
    identicalRunsMoveNothing: baseA.size === baseB.size && [...baseA].every(([id, d]) => baseB.get(id) === d),
  };
  const results = [];
  for (let t = 0; t < trials; t++) {
    const cellId = weights[Math.floor(r.next() * weights.length)].id;
    results.push(localizeOne(net, env, sample, cellId, { eps }));
  }
  const pass = controls.identicalRunsMoveNothing && results.every((x) => x.ok && x.first === x.expectedFirst);
  const summary = {
    trials,
    seed,
    eps,
    cellsHit: [...new Set(results.map((x) => x.cell))],
    slicesExact: results.filter((x) => x.ok).length,
    firstChangedIsConsumer: results.filter((x) => x.first === x.expectedFirst).length,
    totalChangedCells: results.reduce((a, x) => a + x.changed.length, 0),
  };
  return { results, controls, pass, summary };
}

// ── CLI: node src/localize.mjs — run the battery and print the table ──────────────
// (mirrors cellgraph's test_findfault.py output style: one row per perturbation,
// a verdict line, exit code 0 only if every pin held)
export function formatReport(battery) {
  const lines = [];
  lines.push(`  perturbing one weight CELL at a time (every element +${battery.summary.eps}); asking WHICH digests moved\n`);
  lines.push('  weight    first moved  expected   changed  slice  verdict');
  for (const x of battery.results) {
    const ok = x.ok && x.first === x.expectedFirst;
    lines.push(`  ${x.cell.padEnd(9)} ${String(x.first).padEnd(12)} ${String(x.expectedFirst).padEnd(10)} ${String(x.changed.length).padStart(7)} ${String(x.slice.length).padStart(6)}  ${ok ? 'ok' : 'MISMATCH'}`);
    if (!x.ok) {
      if (x.missing.length) lines.push(`     MISSING (digest blind to a real dependency): ${x.missing.join(', ')}`);
      if (x.extra.length) lines.push(`     EXTRA (digest moved without a data edge): ${x.extra.join(', ')}`);
    }
  }
  lines.push('');
  lines.push(`  controls: identical-runs-move-nothing = ${battery.controls.identicalRunsMoveNothing}`);
  lines.push(`  summary : cells hit ${battery.summary.cellsHit.join(',')}; ${battery.summary.slicesExact}/${battery.summary.trials} slices exact; ${battery.summary.firstChangedIsConsumer}/${battery.summary.trials} first-changed = consumer`);
  lines.push('');
  const extraFree = battery.results.every((x) => x.extra.length === 0);
  let verdict;
  if (battery.pass) {
    verdict = 'localization is exact — the changed-digest set IS the downstream slice';
  } else if (extraFree) {
    verdict = 'exact modulo value-invariant cells (named above) — soundness holds: no digest moved without a data edge';
  } else {
    verdict = 'LOCALIZATION FAILED — a digest moved without a data edge; the witness does not match the topology';
  }
  lines.push(`  VERDICT: ${verdict}`);
  return lines.join('\n');
}

export async function main() {
  const { buildAttention, loadNet, initEnv } = await import('./attncells.mjs');
  const { makeTaskData } = await import('./train.mjs');
  const net = loadNet(buildAttention({ vocab: 8, len: 6, dim: 16 }));
  const env = initEnv(net, 7);
  const [sample] = makeTaskData({ task: 'copy', vocab: 8, len: 6, n: 16, seed: 7 });
  const battery = localizeBattery(net, env, sample, { trials: 10, seed: 7, eps: 0.5 });
  console.log(formatReport(battery));
  process.exitCode = battery.pass ? 0 : 1;
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  await main();
}
