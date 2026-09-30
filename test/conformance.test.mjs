// test/conformance.test.mjs — the node side of the cross-runtime conformance pair
// opened by SuperInstance/quilt-attention#1 ("11/11 array digests agree; scalarSha
// is not portable"). The v0.1.0 scalarSha hashed Buffer.from(hex,'hex').toString(
// 'latin1'), which sha256hex re-encoded as UTF-8 — deterministic inside Node, but
// a different preimage than any independent reading of "sha256 over the canonical
// bytes". The fix hashes the legible tagged string `f64|8|<f64hex(x)>`; this suite
// pins node's implementation to the shared fixture that the Python reference
// (test/conformance.py, runnable with plain python3) generated and agrees on.
//
//   fixture:   test/scalar-fixture.json          (32 adversarial scalars + 3 non-finite sentinels)
//   expected:  test/scalar-expected-digests.txt  (the cross-runtime agreement of record)
//   python:    python3 test/conformance.py
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { scalarSha, f64hex, DTYPE_TAG } from '../src/attncells.mjs';

const fixture = JSON.parse(readFileSync(new URL('./scalar-fixture.json', import.meta.url), 'utf8'));
const expected = new Map(
  readFileSync(new URL('./scalar-expected-digests.txt', import.meta.url), 'utf8')
    .trim().split('\n')
    .map((line) => {
      const [name, sha] = line.trim().split(/\s+/);
      return [name, sha];
    }),
);

test(`C1: scalarSha matches the Python reference on all ${fixture.scalars.length} fixture scalars (parse → bytes → digest)`, () => {
  assert.equal(expected.size, fixture.scalars.length, 'expected-digests file and fixture disagree on size');
  for (const s of fixture.scalars) {
    const x = Number(s.text); // node's own parse of the shared decimal string
    assert.ok(Number.isFinite(x), `${s.name}: Number("${s.text}") is not finite`);
    assert.equal(f64hex(x), s.f64hex, `${s.name}: node's f64 bytes differ from the fixture pin`);
    assert.equal(scalarSha(x), expected.get(s.name), `${s.name}: digest differs from the cross-runtime list`);
  }
});

test('C2: non-finite scalars are rejected fail-closed, same as the Python reference', () => {
  for (const t of fixture.nonfinite_must_throw) {
    assert.throws(() => scalarSha(Number(t)), TypeError, `scalarSha(Number("${t}")) must throw`);
  }
});

test('C3: -0 and 0 hash identically (the documented JSON round-trip normalization)', () => {
  assert.equal(f64hex(-0), f64hex(0));
  assert.equal(scalarSha(-0), scalarSha(0));
  assert.equal(scalarSha(-0), expected.get('zero'));
});

test('C4: the preimage is exactly the documented tagged string, hashed as UTF-8', () => {
  // independent golden vectors, not derived from the expected file
  const vectors = [['1', '3ff0000000000000'], ['0.1', '3fb999999999999a'], ['-1.5e-8', 'be501b2b29a4692b']];
  for (const [text, hex] of vectors) {
    assert.equal(scalarSha(Number(text)), createHash('sha256').update(`${DTYPE_TAG}|8|${hex}`, 'utf8').digest('hex'), `golden vector ${text}`);
  }
});
