#!/usr/bin/env python3
"""Hash-bound complete Round23 source invoice; CPU arithmetic only."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

BASE_REL = 'results/packing-swarm/round22/invoice.json'
BASE_SHA = '56f52e46d7c613ff33317e406b900a4d663dcb50e9f689bae28cbd0fe5e8e662'
BASE_BYTES = 21824942
TOOLS_REL = 'method/packing-swarm/tools/round23'
SHAPES = {'input3': 16480, 'attention3': 14336, 'ffn2': 34816}
LIFECYCLE = ('source.hpp', 'native/family.cuh', 'native/native.cu',
    'native/binding.hpp', 'native/generation.hpp', 'native/session.cuh',
    'native/stage.cuh', 'native/views.cuh', 'transport.hpp', 'native/model.hpp',
    'native/packed.hpp', 'native/effects.cuh', 'native/worker.cu')
INHERITED_FIELDS = ('composed_whole_controller', 'E_D', 'inherited_whole_controller',
    'inherited_receiver_capacity', 'original_A8_INT8_endpoint_contract', 'custody_cost',
    'inherited_native_ABI', 'physical_unsets', 'inherited_cost_scope')

class InvoiceError(Exception):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail


def require(condition, code, detail):
    if not condition:
        raise InvoiceError(code, detail)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(root):
    p = root / BASE_REL
    require(p.stat().st_size == BASE_BYTES and sha(p) == BASE_SHA,
        'inherited_invoice_mismatch', str(p))
    return json.loads(p.read_text())


def renamed(s):
    return s.replace('round22', 'round23').replace('paired22', 'paired23').replace('native22', 'native23')


def native_contract(root, inherited_root, old):
    tools = root / TOOLS_REL
    oldtools = inherited_root / 'method/packing-swarm/tools/round22'
    pins = {e['path']: e['SHA256'] for e in old['source']}
    old_compiler = inherited_root/'archive/packing-swarm/round22/compiler1/sources.json'
    old_receipt_pins = {e['path']:e['SHA256'] for e in old['compiler_cost']['receipts']}
    require(sha(old_compiler) == old_receipt_pins[str(old_compiler.relative_to(inherited_root))],
        'inherited_source_mismatch', 'compiled lifecycle closure')
    old_bound = json.loads(old_compiler.read_text())['sources']
    for path, digest in old_bound.items():
        marker = 'method/packing-swarm/tools/round22/'
        if marker in path:
            relative = marker + path.split(marker, 1)[1]
            if relative in pins:
                require(pins[relative] == digest, 'inherited_source_mismatch', relative)
            pins[relative] = digest
    sources = []
    for name in LIFECYCLE + ('native/paired.cuh',):
        before = oldtools / name
        prior = 'method/packing-swarm/tools/round22/' + name
        require(sha(before) == pins[prior], 'inherited_source_mismatch', prior)
        p = tools / name
        sources.append(dict(path=str(p.relative_to(root)), SHA256=sha(p)))
        if name != 'native/paired.cuh':
            require(p.read_text() == renamed(before.read_text()),
                'lifecycle_change_requires_repricing', name)
    before = renamed((oldtools / 'native/paired.cuh').read_text())
    after = (tools / 'native/paired.cuh').read_text()
    helper = '__device__ __forceinline__ float ordered_fold(float d,float scale,float acc) {\n float out;\n asm volatile("fma.rn.f32 %0,%1,%2,%3;" : "=f"(out) : "f"(d),"f"(scale),"f"(acc));\n return out;\n}\n'
    if helper in after:
        require(after.count(helper) == 1 and after.count('=ordered_fold(') == 8,
            'source_dictionary_mismatch', 'ordered fold helper/count')
        fold_variant = 'volatile_PTX_fma_rn_f32'
        after = after.replace(helper, '').replace('=ordered_fold(', '=fmaf(')
    else:
        require('ordered_fold' not in after, 'source_dictionary_mismatch', 'unknown fold helper')
        fold_variant = 'CUDA_fmaf'
    start = '  unsigned a[MT][2][4];float wsc[MT][2];'
    end = '  __syncwarp();if(lane==0)rn::mbar_arrive(&empty[s]);'
    oldpieces = before.split(start)
    require(len(oldpieces) == 3, 'source_dictionary_mismatch', 'two original consumer blocks')
    newstart = '#pragma unroll\n  for(unsigned q=0;q<WG;q++){\n   unsigned a[2][2][4];float wsc[2][2];'
    newpieces = after.split(newstart)
    require(len(newpieces) == 3, 'source_dictionary_mismatch', 'two serialized consumer blocks')
    for i in range(1, 3):
        require(end in oldpieces[i] and end in newpieces[i],
            'source_dictionary_mismatch', 'consumer tail')
        oldpieces[i] = '<SERIALIZED_CONSUMER>' + end + oldpieces[i].split(end, 1)[1]
        body, tail = newpieces[i].split(end, 1)
        tokens = ('const uint4 packed=*(const uint4*)(st+(gw+q)*G::CW+lane*16);',
            'unsigned*o=a[mm][h]', 'wsc[0][0]=s01.x;wsc[0][1]=s01.y;wsc[1][0]=s23.x;wsc[1][1]=s23.y;',
            'for(unsigned pass=0;pass<2;pass++)', 'for(unsigned n=0;n<NT;n++)',
            'rn::ldsm4(b0,b1,b2,b3,as+row*64+((c^((row>>1)&3))<<4));',
            'const uint2 bs=*(const uint2*)(ssc+(n*8+gid)*8);',
            'for(unsigned mm=0;mm<2;mm++)', 'const unsigned m=q*2+mm;',
            'rn::mma_fp4(d,a[mm][0],b0,b1,bs.x,zero);rn::mma_fp4(d,a[mm][1],b2,b3,bs.y,d);',
            'acc[pass][m][n][0]=fmaf(d[0],wsc[mm][0],acc[pass][m][n][0]);',
            'acc[pass][m][n][3]=fmaf(d[3],wsc[mm][1],acc[pass][m][n][3]);')
        require(all(t in body for t in tokens), 'source_dictionary_mismatch', 'serialized consumer ' + str(i))
        require(body.count('rn::ldsm4(') == 1 and body.count('rn::mma_fp4(') == 2
            and body.count('fmaf(') == 4, 'source_dictionary_mismatch', 'consumer operation inventory')
        newpieces[i] = '<SERIALIZED_CONSUMER>' + end + tail
    require(''.join(oldpieces) == ''.join(newpieces),
        'source_dictionary_mismatch', 'producer/geometry/accumulator/epilogue changed')
    p = tools / 'native/generate.py'
    sources.append(dict(path=str(p.relative_to(root)), SHA256=sha(p)))
    return dict(state='matched_source_dictionary', unchanged_lifecycle_files=len(LIFECYCLE),
        producer_geometry_accumulator_epilogue='exact after named round/namespace selectors',
        consumer='group outer; sixteen native operand words/four halfscale floats; both independent pass/n/mm folds',
        complete_native_SHA256=sha(tools / 'native/paired.cuh'), fold_variant=fold_variant, sources=sources,
        refinement='source contract and arithmetic, not concrete CUDA refinement')


def geometry(prior, wg):
    g = copy.deepcopy(prior)
    g.update(serialized_groups=True, weight_native_operand_words_per_group=16,
        live_weight_native_operand_words_per_consumer_thread=16,
        live_halfscale_floats_per_consumer_thread=4,
        prior_live_weight_native_operand_words_per_consumer_thread=16*wg,
        prior_live_halfscale_floats_per_consumer_thread=4*wg,
        source_operand_liveness_bytes=16*4+4*4,
        prior_source_operand_liveness_bytes=(16*4+4*4)*wg,
        physical_register_allocation='unset',
        liveness_contract='Source lexical scope, not an assertion that compiler does not hoist/unroll or retain groups; accumulator128/64 floats unchanged.')
    return g


def requests(prior, wg):
    r = copy.deepcopy(prior)
    r['geometry'] = geometry(prior['geometry'], wg)
    r['shared_q_ldmatrix_warp_instructions'] *= wg
    r['shared_scale_fragment_lane_read_bytes'] *= wg
    r['shared_q_ldmatrix_register_requested_bytes'] = r['shared_q_ldmatrix_warp_instructions']*32*4*4
    fw, folds, ctas = r['geometry']['consumer_warps'], 40, r['CTA_count']
    r.update(source_group_iterations=folds*ctas*fw*wg,
        source_group_pass_iterations=2*folds*ctas*fw*wg,
        source_group_pass_n_iterations=2*folds*ctas*fw*wg*4,
        binary_native_operand_constructions_lane_calls=folds*ctas*fw*wg*32*16,
        half2_to_float2_lane_conversions=folds*ctas*fw*wg*32*2,
        source_q_scale_address_recompositions=2*folds*ctas*fw*wg,
        source_loop_contract='Counts in the explicit unrolled source schedule; address/loop instruction lowering and eliminations are compiler evidence, not assumed instruction charges.',
        shared_operand_load_contract='Each group repeats both pass/n q+scale fragment loads; fullCTA global staging and weight/halfscale loads unchanged.',
        source_dynamic_spill_request_bytes='unset')
    return r


def compiler_cost(root, receipt):
    if receipt is None:
        return dict(state='pending', owner='Round23 native compiler',
            next_action='Supply final source-bound compiler.json and archive compiler1 receipts; rerun generator with --compiler.',
            GPU_execution=False)
    final = json.loads(receipt.read_text())
    require(final['state'] == 'compiled_source_only_zero_GPU_guards' and final['GPU_execution'] is False,
        'compiler_receipt_invalid', 'state or GPU flag')
    directories = [p.parent for p in (root / 'archive/packing-swarm/round23').glob('compiler*/plan.json')
        if json.loads(p.read_text())['source_closure_SHA256'] == final['source_closure_SHA256']]
    require(len(directories) == 1, 'compiler_receipt_invalid', 'unique closure-bound compiler directory')
    directory = directories[0]
    plan = json.loads((directory / 'plan.json').read_text())
    bound = json.loads((directory / 'sources.json').read_text())
    require(final['source_closure_SHA256'] == plan['source_closure_SHA256'],
        'compiler_source_mismatch', 'closure')
    for path, digest in bound['sources'].items():
        p = Path(path)
        if '/method/packing-swarm/' in path and '/work/clones/' in path:
            p = root / ('method/packing-swarm/' + path.split('/method/packing-swarm/', 1)[1])
        elif not p.is_absolute():
            p = root / p
        require(sha(p) == digest, 'compiler_source_mismatch', path)
    require('O1' in plan['host_optimization'], 'compiler_flags_mismatch', 'hostO1')
    require('O3' in plan['device_optimization'] and 'no_fastmath' in plan['device_optimization'],
        'compiler_flags_mismatch', 'deviceO3/no_fastmath')
    arms = {}
    for arm, symbol in [('candidate', '_ZN8paired239candidateENS_4ArgsE'),
            ('strongest_FW8', '_ZN8paired2313strongest_fw8ENS_4ArgsE')]:
        resource, kernel = final['resources'][symbol], final['kernels'][symbol]
        require(resource['dynamic_shared_source_bytes'] == 74112,
            'compiler_resource_mismatch', symbol)
        arms[arm] = dict(symbol=symbol, **resource, **kernel,
            dynamic_spill_request_bytes='unset', occupancy='unset', residency='unset', native_time='unset')
    evidence = []
    for name in ('plan.json', 'sources.json', 'phase-4.log', 'resources.log', 'paired.sass'):
        p = directory / name
        evidence.append(dict(path=str(p.relative_to(root)), SHA256=sha(p)))
    require(sha(directory / 'paired.sass') == final['SASS_SHA256'],
        'compiler_receipt_invalid', 'SASS hash')
    resources_text = (directory/'resources.log').read_text()
    ptxas_text = (directory/'phase-4.log').read_text()
    for arm in arms.values():
        symbol = arm['symbol']
        m = re.search(r'Function '+re.escape(symbol)+r':\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+)', resources_text)
        require(m is not None, 'compiler_receipt_invalid', 'ELF resources '+symbol)
        arm['ELF_reported_shared_bytes'] = int(m[3])
        require(int(m[1]) == arm['registers_per_thread'] and int(m[2]) == arm['stack_frame_bytes'],
            'compiler_resource_mismatch', symbol)
        m = re.search(r'Function properties for '+re.escape(symbol)+r'\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s+: Used (\d+) registers', ptxas_text)
        require(m is not None and tuple(map(int,m.groups())) == (arm['stack_frame_bytes'],
            arm['ptxas_spill_store_bytes'],arm['ptxas_spill_load_bytes'],arm['registers_per_thread']),
            'compiler_resource_mismatch', 'ptxas '+symbol)
    history_dir = root / 'archive/packing-swarm/round23/compiler1'
    historical = json.loads((history_dir/'compiler.json').read_text())
    snapshot = json.loads((history_dir/'source-snapshot.json').read_text())
    historical_sources = json.loads((history_dir/'sources.json').read_text())['sources']
    require(set(snapshot) == set(historical_sources), 'compiler_source_mismatch', 'historical complete snapshot')
    for path, entry in snapshot.items():
        require(entry['SHA256'] == historical_sources[path]
            and sha(history_dir/entry['snapshot_relative']) == historical_sources[path],
            'compiler_source_mismatch', 'historical '+path)
    historical_plan = json.loads((history_dir/'plan.json').read_text())
    require(historical['GPU_execution'] is False and historical['source_closure_SHA256'] == historical_plan['source_closure_SHA256'],
        'compiler_receipt_invalid', 'historical closure')
    history = dict(state='sealed_historical_attempt', variant='serialized_CUDA_fmaf',
        resources=historical['resources'], kernels=historical['kernels'],
        source_closure_SHA256=historical['source_closure_SHA256'],
        complete_source_snapshot_count=len(snapshot),
        receipts=[dict(path=str((history_dir/name).relative_to(root)),SHA256=sha(history_dir/name))
            for name in ('compiler.json','sources.json','source-snapshot.json','plan.json','phase-4.log','resources.log','paired.sass')],
        conclusion='Attempt1 and final explicit per-fold RN retirements retain same register/stack/spill diagnostics; group lexical shortening alone did not remove compiled spills.', GPU_execution=False)
    return dict(state='set', arms=arms, historical_attempt=history, compiler=final['compiler'],
        source_closure_SHA256=final['source_closure_SHA256'], binary_SHA256=final['binary_SHA256'],
        cubin_SHA256=final['cubin_SHA256'], SASS_SHA256=final['SASS_SHA256'],
        receipt=dict(path=str(receipt.relative_to(root)), SHA256=sha(receipt)),
        evidence=evidence, GPU_execution=False,
        scope='Compiled resources/static sites only; no native words, executed SASS, spill bytes, cache, occupancy or time.')


def inherited_ref(pointer):
    return dict(state='hash_bound_inherited_section', document=BASE_REL, SHA256=BASE_SHA,
        JSON_pointer=pointer, resolution='invoice.py --expand verifies exact document then reconstructs complete section')


def full64(old):
    routes = []
    for row in old['full64_paired_routes']:
        families = []
        for family in row['families']:
            f = {k: family[k] for k in ('layer', 'region', 'family')}
            f.update(candidate=requests(family['candidate'], 2),
                strongest=requests(family['strongest'], 1))
            families.append(f)
        totals = {}
        for arm in ('candidate', 'strongest'):
            keys = tuple(row['totals'][arm]) + ('binary_native_operand_constructions_lane_calls',
                'half2_to_float2_lane_conversions', 'source_group_pass_n_iterations',
                'shared_weight_fragment_lane_read_bytes', 'shared_halfscale_fragment_lane_read_bytes',
                'shared_q_ldmatrix_register_requested_bytes')
            totals[arm] = {k: sum(f[arm][k] for f in families) for k in keys}
        routes.append(dict(rows=row['rows'], families=families, totals=totals,
            sum_features=row['sum_features'], retained_control_contract=row['retained_control_contract']))
    return routes


def report(root, inherited_root, receipt):
    old = load_base(inherited_root)
    contract = native_contract(root, inherited_root, old)
    regions = []
    for i, prior in enumerate(old['one_region_configurations']):
        regions.append(dict(family=prior['family'], rows=prior['rows'],
            standalone_paired_family=prior['standalone_paired_family'],
            standalone_supported_edge_routes=prior['standalone_supported_edge_routes'],
            inherited_round21_region=inherited_ref('/one_region_configurations/'+str(i)+'/inherited_round21_region'),
            paired_candidate=requests(prior['paired_candidate'], 2),
            equally_strengthened_FW8=requests(prior['equally_strengthened_FW8'], 1),
            round22_validation_delta=prior['round22_validation_delta'],
            round23_lifecycle_delta=dict(state='no_change_source_bound', device_bytes=0, host_bytes=0,
                Frozen_bytes=0, D2H_bytes=0, CUDA_dispatches=0, explanation='Replace22 consumer kernel, not append extra23 validation arms; all22 controls and observations retained.')))
    routes = full64(old)
    data = dict(state='static_exact_source_request_arithmetic', CPU_only=True, GPU_execution=False,
        source_contract=contract, compiler_cost=compiler_cost(root, receipt),
        generator_SHA256=sha(Path(__file__)), new_packing_bound=False,
        inherited_source=dict(path=BASE_REL, SHA256=BASE_SHA, bytes=BASE_BYTES,
            stable_custody_hint='~/work/packing-mma-round22-evidence/snapshot',
            contract='Exact complete22 document retained, not a delta-only invoice; failed dependency/hash/closure yields typed error. --expand emits all inherited and current sections.'),
        one_region_configurations=regions,
        full64_paired_routes=[dict(rows=r['rows'], totals=r['totals'], sum_features=r['sum_features'],
            family_requests=dict(state='reconstructible', rule='48input3+16attention3+64ffn2; requests() transforms each exact pinned22 full64 family/row record, including B2..7; layer order unchanged')) for r in routes],
        paired_geometry_comparison=dict(candidate=regions[0]['paired_candidate']['geometry'],
            equally_strengthened_FW8=regions[0]['equally_strengthened_FW8']['geometry']),
        retained_harness_complete_totals=[dict(fresh=r['fresh'],
            raw_generation_requests=r['raw_generation_requests'],
            conditional_all_common_generation_requests=r['conditional_all_common_generation_requests'],
            source_success_assumed_layer_local_raw_peak=r['source_successful_early_release_generation_raw_peak'],
            D2H_projection_bytes=r['D2H_projection_bytes'],
            retained_Stage_and_family_host_bytes=r['retained_Stage_and_family_host_bytes'],
            Frozen_source_image_bytes=r['Frozen_source_image_bytes'],
            exact_validation_release_calls_per_full_success=r['exact_validation_release_calls_per_full_success'],
            max_layer_handles=max(p['exact_validation_handle_count'] for p in r['early_release_layer_peaks']),
            prior_round22_additions=r['round22_additions']) for r in old['composed_whole_controller']['rows']],
        complete_configuration_count=24,
        source_conclusion='Weight/global q/halfscale/global scale staging, launches, folds, aggregate accumulator bytes and complete lifecycle match. Source native weight/scales lexical liveness halves only forWG2; candidate q/scale fragment requests double against22 and now equal WG1 strongest aggregates. Candidate128 vs64 accumulator floats/thread persists.',
        remaining_unsets=['native CUDA words/faults/time', 'dynamic spill traffic', 'concrete primitive/controller refinement',
            'physical allocation/cache/residency/occupancy', 'tensor-wide B8/B32 controller', 'model/logit/task/HTTP'])
    for key in INHERITED_FIELDS:
        data[key] = inherited_ref('/'+key)
    compiler = data['compiler_cost']
    if compiler['state'] == 'pending':
        data['nomination'] = dict(state='pending_compiler', owner='native coordinator')
    else:
        a, b = compiler['arms']['candidate'], compiler['arms']['strongest_FW8']
        spilled = a['ptxas_spill_store_bytes'] > 0 or a['ptxas_spill_load_bytes'] > 0
        data['nomination'] = dict(state='none_cost_sidecar',
            remaining_loss='candidate compiled spills against strongest' if spilled else
                'matched recurring bytes/launches and accumulator asymmetry remain; coordinator judges complete metric tradeoffs',
            candidate_registers=a['registers_per_thread'], strongest_registers=b['registers_per_thread'])
    checks(data, old)
    return data, old, routes


def checks(data, old):
    assert len(data['one_region_configurations']) == 9
    for i, region in enumerate(data['one_region_configurations']):
        a, b = region['paired_candidate'], region['equally_strengthened_FW8']
        prior = old['one_region_configurations'][i]['paired_candidate']
        assert a['geometry']['accumulator_floats_per_consumer_thread'] == 128
        assert b['geometry']['accumulator_floats_per_consumer_thread'] == 64
        assert a['geometry']['dynamic_shared_bytes'] == b['geometry']['dynamic_shared_bytes'] == 74112
        assert a['shared_q_ldmatrix_warp_instructions'] == b['shared_q_ldmatrix_warp_instructions'] == 2*prior['shared_q_ldmatrix_warp_instructions']
        assert a['shared_scale_fragment_lane_read_bytes'] == b['shared_scale_fragment_lane_read_bytes'] == 2*prior['shared_scale_fragment_lane_read_bytes']
        for key in ('weight_and_halfscale_global_to_shared_requested_bytes', 'q_and_scale_global_to_shared_requested_bytes',
                'reader_launches', 'executed_warp_MMA_instructions', 'ordered_fold_FMA_lane_updates',
                'shared_weight_fragment_lane_read_bytes', 'shared_halfscale_fragment_lane_read_bytes'):
            assert a[key] == b[key] == prior[key]
        assert region['standalone_paired_family'] == old['one_region_configurations'][i]['standalone_paired_family']
    assert len(data['full64_paired_routes']) == 8 and data['complete_configuration_count'] == 24
    for r in data['full64_paired_routes']:
        a, b = r['totals']['candidate'], r['totals']['strongest']
        assert a['reader_launches'] == b['reader_launches'] == 128
        assert a['CTA_count'] == b['CTA_count'] == 12720
        assert a['weight_and_halfscale_global_to_shared_requested_bytes'] == b['weight_and_halfscale_global_to_shared_requested_bytes'] == 2339020800
        assert a['q_and_scale_global_to_shared_requested_bytes'] == b['q_and_scale_global_to_shared_requested_bytes'] == 2344550400
        assert a['shared_q_ldmatrix_warp_instructions'] == b['shared_q_ldmatrix_warp_instructions'] == 32563200
        assert a['shared_q_ldmatrix_register_requested_bytes'] == b['shared_q_ldmatrix_register_requested_bytes'] == 16672358400
        assert a['shared_scale_fragment_lane_read_bytes'] == b['shared_scale_fragment_lane_read_bytes'] == 8336179200
        assert 2*a['consumer_empty_barrier_arrival_calls'] == b['consumer_empty_barrier_arrival_calls']
    for row in data['retained_harness_complete_totals']:
        assert row['exact_validation_release_calls_per_full_success'] == 14573
        assert row['max_layer_handles'] == 248


def expand(data, old, routes):
    result = copy.deepcopy(data)
    for key in INHERITED_FIELDS:
        result[key] = copy.deepcopy(old[key])
    for i, region in enumerate(result['one_region_configurations']):
        region['inherited_round21_region'] = copy.deepcopy(old['one_region_configurations'][i]['inherited_round21_region'])
    result['full64_paired_routes'] = routes
    result['resolution'] = dict(state='complete_inherited_and_current_invoice', inherited_SHA256=BASE_SHA,
        unchanged_harness_contract='Owning-round labels preserved inside historical charges; complete amounts current23 unchanged by exact native lifecycle binding.')
    return result


def markdown(data):
    lines = ['# Round23 — group-local native operand lifetime invoice', '',
        'CPU/source requests and compiled resources only. Standalone and retained harness are separate complete prices.', '',
        '## Native source map', '', data['source_conclusion'], '',
        'Both arms: eight32-feature groups/256features perCTA, TOKW32, K5120, forty ascendingK128 folds per independent pass. Stage9216B; eight stages+128B mbarriers+256B globals=74112B dynamic shared. Consumer128/64 accumulator floats/thread, equal65536B accumulator payload/CTA. Serializing weight fragments does not serialize accumulators.', '',
        'Live lexical weight payload candidate32→16 native operand words and8→4 unequal halfscale floats; strongest16→16 and4→4 receives the same group-local transform. Native packing and E/D unchanged. Group/pass/n q+scale shared loads repeat; compiler may unroll/eliminate address or load work, so lexical counts are not dynamic SASS claims.', '',
        '| Family | Rows | CTAs | W+halfscale B | q+scale staging B | candidate/strongest LDSM source calls | candidate/strongest shared scale lane B |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in data['one_region_configurations']:
        a = r['paired_candidate']
        lines.append(f"| {r['family']} | {r['rows']} | {a['CTA_count']} | {a['weight_and_halfscale_global_to_shared_requested_bytes']} | {a['q_and_scale_global_to_shared_requested_bytes']} | {a['shared_q_ldmatrix_warp_instructions']} | {a['shared_scale_fragment_lane_read_bytes']} |")
    lines += ['', 'Source weight/halfscale fragment loads, binary expansion/half2 conversions, MMA/ordered-FMA work remain equal. Global staging includes padded32 loaded rows from physically64-row tapes even atB1/B8. GDN has three real tail groups but complete padded eight-group compute is paid. No extra global q stream is introduced.', '',
        '## Complete standalone price', '',
        'Each arm owns original source+Norm, one complete two-pass Tape including physical64-row q/s, globals/transformed/Norm/statuses, independent prepared binary weights+original half scales, and16BF output/primary/two-owner bytes. Both launch two producer passes+one paired reader. Candidate prep still pays producer-status D2H/checks; generic strongest prep does not. Complete regional E/D, quantizers, direct preparation/down borrowed consumers and exactA8/INT8/materialized controls are resolved from the pinned inherited sections, not silently dropped.', '',
        '| Family | Rows | Each arm family device B | Candidate direct edge device B | Strongest direct edge device B |',
        '|---|---:|---:|---:|---:|']
    for r in data['one_region_configurations']:
        family, edge = r['standalone_paired_family'], r['standalone_supported_edge_routes']
        if edge['state'] == 'logical_source_requests':
            a = edge['candidate_direct']['combined_device_bytes']
            b = edge['strongest_merged_FW8_direct']['combined_device_bytes']
        elif edge['state'] == 'no_supported_direct_preparation':
            a = b = 'original fixed-layout route'
        else:
            raise InvoiceError('unsupported_edge_state', edge['state'])
        lines.append(f"| {r['family']} | {r['rows']} | {family['candidate_device_requested_bytes']} | {a} | {b} |")
    lines += ['', 'FFN candidate retains its two separately owned preparation/down validation co-readers and four owners; strongest retains original generic continuation. The input direct edge includes the original GDN preparation contract; attention retains its original fixed-layout continuation and its inherited whole-controller charge. These are requested source allocation sizes, not physical capacity/time.', '',
        'Full64 each standalone paired arm:128reader launches,12720CTAs,2339020800B W+halfscale,2344550400B q+scale and101760*rowsB global scales. Each now requests32563200 source warpLDSM (16672358400B lane-register payload) and8336179200B shared scale lane reads. Candidate barrier arrivals/waits remain half of strongest; neither global bytes nor launches improve relative to22.', '',
        '## Complete retained-harness price', '',
        'Round23 replaces the22 consumer schedules, rather than appending another arm. Thirteen source/lifecycle files match22 exactly after named round/namespace selectors; producer/geometry/accumulator/epilogue also match. Thus device/host/Frozen/D2H/CUDA-call/owner/output/lifecycle deltas are exactly zero. This is not a zero total: both new paired arms, original21/current19/materialized20/independent literalFW8/exactA8/INT8 controls and primary reference remain executed and observed.', '',
        'The22 validation additions stay paid:9allocations/36BF device bytes per family,36BF initial+28BF explicit fills,52BF D2H/13calls,40BF host and40BF+4 Frozen bytes,47CUDA calls before release,12pass manifest events/family. Full64 has1152extra handles/14573total/max248perlayer;512extra manifest groups and6016added CUDA calls. Added retained control projection launch counts384, W+halfscale7017062400B and q+scale5861376000B are still retained.', '',
        '| Fresh rows | Complete retained-all device request B | Success-assumed layer-local peak B | Complete D2H projection B | Complete Frozen image B |',
        '|---|---:|---:|---:|---:|']
    for row in data['retained_harness_complete_totals']:
        if row['fresh'] in (1, 8):
            lines.append(f"| {row['fresh']} | {row['raw_generation_requests']} | {row['source_success_assumed_layer_local_raw_peak']} | {row['D2H_projection_bytes']} | {row['Frozen_source_image_bytes']} |")
    lines += ['', 'Every source/Norm/fullstride image, source preservation/end observation, status/fault owners, ring/convpending/bothpending/checkpoint/F16KV/head/logits/argmax, manifest/source/consumer drain, exact spans/views, uncertain backing retention, Frozen/reset/old8-newR/deepclone/D2D/serialization/rename/directoryfsync/whole-release charge remains in the reconstructible complete invoice. All8fresh-row full64 cases and24serialized independent-stream configurations are inherited at exact content. No allocator capacity, cache effect, residency or physical peak is inferred.', '',
        '## Compiler and remaining loss', '']
    c = data['compiler_cost']
    if c['state'] == 'pending':
        lines += ['Compiler evidence pending with native owner. No registers, spills or nomination invented.']
    else:
        a, b = c['arms']['candidate'], c['arms']['strongest_FW8']
        lines += [f"{c['compiler']}: candidate{a['registers_per_thread']}registers/thread,{a['stack_frame_bytes']}B stack,{a['ptxas_spill_store_bytes']}B spill stores/{a['ptxas_spill_load_bytes']}B spill loads; strongest{b['registers_per_thread']}registers,{b['stack_frame_bytes']}B stack,{b['ptxas_spill_store_bytes']}B stores/{b['ptxas_spill_load_bytes']}B loads.", '',
            f"Static SASS sites candidate{a['static_SASS_instructions']}/strongest{b['static_SASS_instructions']}; LDL/STL sites{a['static_local_load_sites']}/{a['static_local_store_sites']} versus{b['static_local_load_sites']}/{b['static_local_store_sites']}. Static shared-load sites{a['static_shared_load_sites']}/{b['static_shared_load_sites']}; ELF shared{a['ELF_reported_shared_bytes']}/{b['ELF_reported_shared_bytes']}B separately from ptxas user static shared. Exact source closure/cubin/SASS hashes and complete opcode/resource receipts are bound in JSON. Static sites and ptxas spill diagnostics are not executed traffic/time.", '',
            data['nomination']['remaining_loss'] + '.', '',
            f"Initial serialized CUDA_fmaf attempt is separately sealed with{c['historical_attempt']['complete_source_snapshot_count']}exact source snapshots and all compiler resources/opcodes. Both attempts retain255candidate/131strongest registers,16/0B stacks and28/0B each spill diagnostic; final explicit volatile fma.rn.f32 preserves each scalar ordered fold without changing fold/tree/count or literal pass additions."]
    lines += ['', '**Unset:** ' + '; '.join(data['remaining_unsets']) + '.', '',
        '## Executable inherited custody', '',
        f"Inherited invoice SHA256 `{BASE_SHA}` ({BASE_BYTES}B). Compact JSON contains exact hash-bound JSON pointers; `--expand` resolves every complete section and128family requests/row without copying22MB into normal delivery. Missing or altered dependency/source/receipt returns typed error, never a zero charge.", '',
        'Replay from admitted programme root: `python3 method/packing-swarm/tools/round23/cost/invoice.py --root . --inherited-root . --out results/packing-swarm/round23 --compiler results/packing-swarm/round23/compiler.json --verify`.', '',
        'Full reconstruction: same root/inherited/compiler arguments, replace `--verify` with `--expand /ABS/PATH/complete-invoice.json`. Pending compiler mode is explicit `--compiler-pending` instead of `--compiler`.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inherited-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    c = parser.add_mutually_exclusive_group(required=True)
    c.add_argument('--compiler', type=Path)
    c.add_argument('--compiler-pending', action='store_true')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--verify', action='store_true')
    action.add_argument('--expand', type=Path)
    args = parser.parse_args()
    try:
        root, inherited_root, out = args.root.resolve(), args.inherited_root.resolve(), args.out.resolve()
        receipt = None if args.compiler_pending else args.compiler.resolve()
        data, old, routes = report(root, inherited_root, receipt)
        text = markdown(data)
        if args.verify:
            require(json.loads((out/'invoice.json').read_text()) == data and (out/'COST.md').read_text() == text,
                'saved_invoice_mismatch', str(out))
        elif args.expand:
            args.expand.write_text(json.dumps(expand(data, old, routes), indent=2)+'\n')
        elif not args.check:
            out.mkdir(parents=True, exist_ok=True)
            (out/'invoice.json').write_text(json.dumps(data, indent=2)+'\n')
            (out/'COST.md').write_text(text)
    except InvoiceError as e:
        print(json.dumps(dict(ok=False, error=e.code, detail=e.detail))); return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(json.dumps(dict(ok=False, error='boundary_input_error', kind=type(e).__name__, detail=str(e)))); return 2
    print(json.dumps(dict(ok=True, state=data['state'], regions=9, full64_rows=8,
        independent_stream_configurations=24, inherited_SHA256=BASE_SHA,
        compiler=data['compiler_cost']['state'], nomination=data['nomination'], GPU_execution=False)))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
