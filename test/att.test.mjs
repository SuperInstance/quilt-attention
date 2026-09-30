// quilt-attention — the six-test battery (node:test, zero deps).
// T1  train to the accuracy bar (reversal, vocab 6, len 4, seed 7, 1200 epochs)
// T2  same seed → byte-identical receipt chains
// T3  hand-rolled backprop vs central finite differences (rel err < 1e-4, 10 params)
// T4  fault localization: changed-digest set == downstream topological slice, 5/5
// T5  a tampered chain fails verify()
// NC1 forward passes over ANY valid topo order give byte-identical digests
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildAttention, loadNet, initEnv, evaluateForward, randomTopoOrder, lcg } from '../src/attncells.mjs';
import { makeTaskData, train, verify, verifyWhy, meanLoss, fullBatchGrad } from '../src/train.mjs';
import { localizeBattery } from '../src/localize.mjs';

const CFG = { task: 'reverse', vocab: 6, len: 4, n: 16, seed: 7 }; // the toy task
const NET = () => loadNet(buildAttention({ vocab: 6, len: 4, dim: 16 }));
const DATA = () => makeTaskData(CFG);
const TRAIN = (epochs = 1200, lr = 0.1) => train(NET(), { data: DATA(), epochs, lr, seed: 7, mode: 'sgd' });

test('T1: reversal reaches acc >= 0.95 within 1200 epochs (seed 7), chain verifies', () => {
  const res = TRAIN();
  const at = res.metrics.findIndex((m) => m.acc >= 0.95);
  assert.ok(at !== -1, `acc never reached 0.95; final ${res.finalAcc}`);
  assert.ok(at <= 1200, `reached 0.95 at epoch ${at}, past budget`);
  assert.ok(res.finalAcc >= 0.95, `final acc ${res.finalAcc} < 0.95`);
  assert.equal(verify(res.chain), true);
  assert.equal(res.chain.length, 1200);
});

test('T2: same seed → byte-identical chains', () => {
  const a = TRAIN(200);
  const b = TRAIN(200);
  assert.equal(JSON.stringify(a.chain), JSON.stringify(b.chain));
  assert.equal(a.tip, b.tip);
});

test('T3: analytic gradients match central finite differences (rel err < 1e-4)', () => {
  const net = NET();
  const env = initEnv(net, 7);
  const data = DATA().slice(0, 4);
  const analytic = fullBatchGrad(net, env, data);
  const params = [];
  for (const w of net.weightCells) {
    const g = analytic.get(w.id);
    for (let i = 0; i < g.length; i++) if (g[i] !== 0) params.push([w.id, i, g[i]]);
  }
  params.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2])); // the 10 strongest responses
  const h = 1e-5;
  let worst = 0;
  for (const [id, i, g] of params.slice(0, 10)) {
    const arr = env[id];
    const keep = arr[i];
    arr[i] = keep + h; const lp = meanLoss(net, env, data);
    arr[i] = keep - h; const lm = meanLoss(net, env, data);
    arr[i] = keep;
    const fd = (lp - lm) / (2 * h);
    const rel = Math.abs(fd - g) / Math.max(1e-8, Math.abs(fd), Math.abs(g));
    assert.ok(rel < 1e-4, `${id}[${i}]: fd=${fd} analytic=${g} rel=${rel}`);
    worst = Math.max(worst, rel);
  }
  assert.ok(params.length >= 10, 'not enough nonzero-gradient params');
});

test('T4: localization — changed set == downstream slice, 5/5 seeded cells', () => {
  const net = NET();
  const env = initEnv(net, 7);
  // Sample data[1]: on some samples a uniform +eps on ALL of Wo shifts every logit of
  // a position by the SAME constant, which max-subtracted cross-entropy is exactly
  // invariant to (documented caveat in src/localize.mjs) — the loss digest then
  // legitimately stays put. data[1] shows the rounding, so the exact pin applies.
  const sample = DATA()[1];
  const battery = localizeBattery(net, env, sample, { trials: 5, seed: 7, eps: 0.5 });
  assert.equal(battery.controls.identicalRunsMoveNothing, true);
  for (const r of battery.results) {
    assert.deepEqual([...new Set(r.changed)].sort(), [...new Set(r.slice)].sort(),
      `${r.cell}: changed ${r.changed} vs slice ${r.slice} (missing ${r.missing}, extra ${r.extra})`);
    assert.equal(r.first, r.expectedFirst, `${r.cell}: first-changed must be the direct consumer`);
  }
  assert.equal(battery.pass, true);
});

test('T5: a tampered chain fails verify()', () => {
  const res = TRAIN(10);
  assert.equal(verify(res.chain), true);
  const evil = structuredClone(res.chain);
  evil[3].loss = 0.0000001; // forge a better loss
  const why = verifyWhy(evil);
  assert.equal(why.ok, false);
  assert.match(why.reason, /sha|loss|seq|prev/);
  assert.equal(verify(evil), false);
});

test('NC1: cell-order independence — any valid topo order, byte-identical digests', () => {
  const net = NET();
  const env = initEnv(net, 7);
  const [sample] = DATA();
  const digests = (order) => {
    const w = evaluateForward(net, env, sample, order, true);
    return new Map(w.map((x) => [x.cell, x.digest])); // per-cell, order-free
  };
  const baseline = digests(undefined);
  const r = lcg(99);
  for (let k = 0; k < 3; k++) {
    const order = randomTopoOrder(net.graph, r);
    assert.notEqual(order.join(','), net.order.join(',')); // the order is REALLY different
    assert.deepEqual(digests(order), baseline, `topo order #${k} changed a digest`);
  }
});
