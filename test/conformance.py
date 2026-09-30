#!/usr/bin/env python3
"""test/conformance.py — the Python side of the cross-runtime conformance pair
opened by SuperInstance/quilt-attention#1 ("11/11 array digests agree; scalarSha
is not portable, same line as quilt-nn lossShaOf").

This file is the independent reference implementation of the spec both repos now
document: a scalar digest is

    sha256(utf8("f64|8|" + f64hex(x)))

with f64hex(x) the 8 big-endian IEEE-754 bytes of x as lowercase hex, -0
normalized to +0, and non-finite inputs rejected fail-closed. The node side is
test/conformance.test.mjs (run by `npm test`); the shared agreement of record is
test/scalar-expected-digests.txt next to the shared test/scalar-fixture.json.

Run: python3 test/conformance.py   (stdlib only)
"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DTYPE_TAG = 'f64'


def f64hex(x):
    x = float(x)
    if x == 0.0:
        x = 0.0  # normalize -0.0 -> +0.0, matching f64hex in both repos
    return struct.pack('>d', x).hex()


def scalar_sha(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f'scalar_sha needs a finite number, got {x!r}')
    return hashlib.sha256(f'{DTYPE_TAG}|8|{f64hex(x)}'.encode('utf-8')).hexdigest()


def main() -> int:
    fixture = json.loads((HERE / 'scalar-fixture.json').read_text(encoding='utf-8'))
    expected = {}
    for line in (HERE / 'scalar-expected-digests.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            name, sha = line.split()
            expected[name] = sha

    digest_ok = 0
    for s in fixture['scalars']:
        x = float(s['text'])  # python's own parse of the shared decimal string
        assert math.isfinite(x), f"{s['name']}: float({s['text']!r}) is not finite"
        assert f64hex(x) == s['f64hex'], f"{s['name']}: f64 bytes differ from the fixture pin"
        d = scalar_sha(x)
        assert d == expected[s['name']], f"{s['name']}: {d} != expected {expected[s['name']]}"
        digest_ok += 1

    throw_ok = 0
    for t in fixture['nonfinite_must_throw']:
        try:
            scalar_sha(float(t))
        except ValueError:
            throw_ok += 1
        else:
            raise AssertionError(f'scalar_sha(float({t!r})) must raise')

    assert f64hex(-0.0) == f64hex(0.0) == '0000000000000000'
    assert scalar_sha(-0.0) == scalar_sha(0.0) == expected['zero']

    print(f"OK [{fixture['spec']}]")
    print(f'  scalar digest agreement : {digest_ok}/{digest_ok}')
    print(f'  non-finite fail-closed  : {throw_ok}/{throw_ok}')
    print('  -0 == 0 normalization   : agree')
    return 0


if __name__ == '__main__':
    sys.exit(main())
