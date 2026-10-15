#!/usr/bin/env python3
"""Seconds-only cost reconstruction and typed refusal witnesses; no GPU paths."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import invoice


def expect_error(fn, code):
    try:
        fn()
    except invoice.InvoiceError as e:
        assert e.code == code, (e.code, code)
        return dict(state='expected_typed_error', error=e.code)
    raise AssertionError('missing_expected_typed_error:'+code)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--inherited-root', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root, inherited = a.root.resolve(), a.inherited_root.resolve()
    data, old, routes = invoice.report(root, inherited, a.compiler.resolve())
    complete = invoice.expand(data, old, routes)
    for key in invoice.INHERITED_FIELDS:
        assert complete[key] == old[key]
    for i, r in enumerate(complete['one_region_configurations']):
        assert r['inherited_round21_region'] == old['one_region_configurations'][i]['inherited_round21_region']
    assert len(complete['composed_whole_controller']['configurations']) == 24
    assert len(complete['full64_paired_routes']) == 8
    assert all(len(r['families']) == 128 for r in complete['full64_paired_routes'])
    for row in complete['full64_paired_routes']:
        for arm in ('candidate','strongest'):
            for key, total in row['totals'][arm].items():
                assert total == sum(f[arm][key] for f in row['families'])
    failures = []
    with tempfile.TemporaryDirectory(prefix='round23-cost-witness-') as tmp:
        t = Path(tmp)
        destination = t / invoice.BASE_REL
        destination.parent.mkdir(parents=True)
        destination.write_text('altered inherited content\n')
        failures.append(expect_error(lambda: invoice.load_base(t), 'inherited_invoice_mismatch'))
        tools = t / invoice.TOOLS_REL
        for name in invoice.LIFECYCLE + ('native/paired.cuh', 'native/generate.py'):
            target = tools/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/invoice.TOOLS_REL/name,target)
        invoice.native_contract(t,inherited,old)
        target = tools/'source.hpp'
        original = target.read_text()
        target.write_text(original+'\n')
        failures.append(expect_error(lambda: invoice.native_contract(t,inherited,old), 'lifecycle_change_requires_repricing'))
        target.write_text(original)
        target = tools/'native/paired.cuh'
        original = target.read_text()
        assert 'fma.rn.f32' in original
        target.write_text(original.replace('fma.rn.f32','fma.rz.f32'))
        failures.append(expect_error(lambda: invoice.native_contract(t,inherited,old), 'source_dictionary_mismatch'))
    encoded = (json.dumps(complete,indent=2)+'\n').encode()
    receipt = dict(state='complete_reconstruction_and_typed_refusals', GPU_execution=False,
        generator_SHA256=invoice.sha(Path(invoice.__file__)), test_SHA256=invoice.sha(Path(__file__)),
        inherited_SHA256=invoice.BASE_SHA, native_source_SHA256=data['source_contract']['complete_native_SHA256'],
        compiler_source_closure_SHA256=data['compiler_cost']['source_closure_SHA256'],
        historical_source_snapshot_count=data['compiler_cost']['historical_attempt']['complete_source_snapshot_count'],
        complete_JSON_bytes=len(encoded), complete_JSON_SHA256=hashlib.sha256(encoded).hexdigest(),
        regions=9, full64_row_cases=8, full64_family_records=1024,
        independent_stream_configurations=24, typed_refusal_witnesses=failures,
        comparison='All inherited sections equal exact pinned22 content; all current full64 totals sum their1024 source request records.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
