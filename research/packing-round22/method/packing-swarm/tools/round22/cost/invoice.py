#!/usr/bin/env python3
"""Exact CPU/source invoice. No CUDA launch, device query, time or residency inference."""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / 'results/packing-swarm/round21/invoice.json'
TOOLS = ROOT / 'method/packing-swarm/tools/round22'
OUT = ROOT / 'results/packing-swarm/round22'
SHAPES = {'input3': 16480, 'attention3': 14336, 'ffn2': 34816}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(fw, wg):
    if (fw, wg) not in ((4, 2), (8, 1)):
        return dict(ok=False, error='unsupported_paired_geometry')
    return dict(ok=True, value=dict(consumer_warps=fw, producer_warps=1,
        threads=32*(fw+1), groups_per_warp=wg, groups_per_CTA=8,
        features_per_CTA=256, token_tile=32, weight_chunk_bytes=512,
        halfscale_chunk_bytes=64, stage_weight_bytes=4096,
        stage_halfscale_bytes=512, stage_q_bytes_per_pass=2048,
        stage_scale_bytes_per_pass=256, stage_bytes=9216, ring_slots=8,
        mbarrier_count=16, mbarrier_bytes=128, shared_global_scale_bytes=256,
        dynamic_shared_bytes=8*9216+128+256,
        accumulator_floats_per_consumer_thread=2*(2*wg)*4*4,
        accumulator_payload_bytes_per_CTA=65536,
        stage_byte_offsets=dict(weight=0,halfscale=4096,q0=4608,s0=6656,q1=6912,s1=8960,end=9216),
        mbarrier_full_byte_offset=73728,mbarrier_empty_byte_offset=73792,
        global_scales_byte_offset=73856,global_scales_bytes=256,
        inline_binary_weight_expansion=True, global_lookup_book_bytes=0,
        shared_lookup_book_bytes=0, physical_register_allocation='unset'))


def requests(F, B, fw, wg):
    if type(F) is not int or F not in SHAPES.values() or type(B) is not int or B not in (*range(1, 9), 32):
        return dict(ok=False, error='unsupported_family_or_rows')
    g = geometry(fw, wg)
    if not g['ok']:
        return g
    groups, folds, ctas = F//32, 40, (F+255)//256
    return dict(ok=True, value=dict(rows=B, features=F, K=5120,
        CTA_count=ctas, CTA_tail_groups=groups%8 or 8, reader_launches=1,
        independent_passes=2, ascending_K128_folds_per_pass=folds,
        physical_q_scale_rows=64, loaded_token_rows_per_CTA=32,
        weight_global_to_shared_requested_bytes=folds*F*16,
        halfscale_global_to_shared_requested_bytes=folds*F*2,
        weight_and_halfscale_global_to_shared_requested_bytes=folds*F*18,
        q_global_to_shared_requested_bytes=2*folds*ctas*2048,
        scale_global_to_shared_requested_bytes=2*folds*ctas*256,
        q_and_scale_global_to_shared_requested_bytes=2*folds*ctas*2304,
        global_scale_global_to_shared_requested_bytes=8*B*ctas,
        producer_bulk_copy_instructions=6*folds*ctas,
        weight_prefetch_instructions_if_enabled=(folds-8)*ctas,
        prefetched_weight_range_requested_bytes_if_enabled=(folds-8)*F*16,
        prefetch_contract='Additional prefetch requests, not a second counted weight stream or a cache hit claim; enabled state is an inherited native dynamic predicate.',
        mbarrier_init_calls=16*ctas, mbarrier_expect_calls=folds*ctas,
        producer_empty_barrier_wait_calls=(folds-8)*ctas,
        consumer_full_barrier_wait_calls=fw*folds*ctas,
        consumer_empty_barrier_arrival_calls=fw*folds*ctas,
        warp_sync_calls=fw*folds*ctas, CTA_sync_calls=ctas,
        mbarrier_init_fence_calls=ctas,
        pdl_wait_source_calls=(1+fw)*ctas, pdl_launch_source_calls=fw*ctas,
        barrier_wait_retry_iterations='unset',
        shared_weight_fragment_lane_read_bytes=folds*ctas*fw*wg*32*16,
        shared_halfscale_fragment_lane_read_bytes=folds*ctas*fw*wg*32*8,
        shared_q_ldmatrix_warp_instructions=2*folds*ctas*fw*4,
        shared_scale_fragment_lane_read_bytes=2*folds*ctas*fw*32*4*8,
        shared_global_scale_lane_read_bytes=2*ctas*fw*8*B*4,
        shared_global_scale_write_bytes=256*ctas,
        tail_unloaded_weight_groups=8*ctas-groups,
        tail_contract='Producer copies only real groups; consumers execute all padded warp fragments before output guards. Unloaded tail shared values never own a valid output; padded compute still charged.',
        executed_warp_MMA_instructions=2*folds*ctas*fw*(2*wg)*4*2,
        ordered_fold_FMA_lane_updates=2*folds*ctas*fw*32*(2*wg)*4*4,
        output_prior_read_bytes=4*B*F, primary_snapshot_write_bytes=4*B*F,
        final_output_write_bytes=4*B*F, output_read_plus_write_requested_bytes=12*B*F,
        owner_write_bytes=8*B*F, primary_snapshot_device_bytes=4*B*F,
        output_and_two_pass_owners_and_snapshot_device_bytes=16*B*F,
        geometry=g['value'], native_words='unset', native_time='unset',
        cache_effective_traffic='unset', occupancy='unset'))


def validation_delta(F, B):
    ctas=(F+255)//256
    return dict(rows=B,features=F,
        paired_image_float_arrays=4,paired_image_owner_arrays=4,
        original_primary_reference_arrays=1,
        additional_device_allocations=9, additional_device_requested_bytes=36*B*F,
        additional_allocation_initial_fill_bytes=36*B*F,
        additional_explicit_fill_bytes=28*B*F,
        additional_D2H_bytes=52*B*F, additional_D2H_calls=13,
        additional_retained_family_host_bytes=40*B*F,
        additional_original_primary_transient_host_bytes=4*B*F,
        prolonged_original_FW8_final_transient_host_bytes=4*B*F,
        run_new_transient_host_upper_bytes=8*B*F,
        prolonged_final_contract='Original all vector is copied to direct_control instead of moved to control_output; its lifetime now runs through the remainder of family run. Pay the existing allocation prolonged4BF alongside4BF original-primary local; preservation adds16BF separately.',
        additional_preservation_transient_host_bytes=16*B*F,
        additional_Frozen_source_bytes=40*B*F+4,
        additional_Frozen_append_copy_payload_bytes=40*B*F+4,
        additional_Frozen_append_copy_read_plus_write_bytes=2*(40*B*F+4),
        enlarged_copy_contract='Every subsequent LayerInput/Frozen/provenance/old-new/deepclone/serialization source-image copy pays the enlarged complete byte image; full composed image sizes and clone copy deltas are supplied. Growth relocation counts/vector/control-block capacities remain unset.',
        native_paired_image_pointer_member_bytes=2*4*8,
        added_direct_host_vector_fields=2,added_paired_word_vector_fields=8,
        source_member_control_block_and_allocator_bytes='unset',
        additional_validation_handle_bytes=9*80,
        additional_validation_native_pointer_bytes=9*8,
        added_successful_CUDA_calls=dict(allocate=9,initial_fill=9,extra_fill=7,
            attribute=3,launch=3,fence=3,D2H=13),
        added_successful_CUDA_calls_without_release=47,
        additional_projection_manifest_groups=4,
        additional_projection_start_launched_observed_calls=12,
        paired_primary_and_final_comparison_payload_bytes=16*B*F,
        additional_end_preservation_comparison_payload_bytes=16*B*F,
        additional_owner_status_words_checked=4*B*F,
        additional_host_vector_copy_payload_bytes=16*B*F,
        additional_host_vector_copy_read_plus_write_bytes=32*B*F,
        additional_reader_launches=3,
        retained_validation_contract='Original21 direct candidate/control final outputs remain and compare to the two paired arms. The extra unmodifiedFW8 primary reference is a real separately allocated/fill/launched/fence/D2H observation. All nine arrays remain through whole-family validation release.',
        original_primary_reference=dict(reader_launches=1,
            weight_and_halfscale_requested_bytes=40*F*18,
            q_scale_requested_bytes=40*ctas*2304,
            output_zero_fill_bytes=4*B*F,
            output_read_plus_write_requested_bytes=8*B*F,
            D2H_bytes=4*B*F,
            original_global_scale_lane_requested_bytes=B*F),
        validation_added_weight_and_halfscale_requests=3*40*F*18,
        validation_added_q_scale_requests=5*40*ctas*2304,
        standalone_consequence='Original primary reference and inherited controls are validation charges, not required extra recurring standalone reader launches; both standalone paired arms use one reader launch and equally observed two passes.',
        physical_capacities='unset')


def full64_paired_routes(B):
    families=[]
    for layer in range(64):
        for region,kind in [('layer_input','attention3' if layer%4==3 else 'input3'), ('post_attention','ffn2')]:
            F=SHAPES[kind]
            families.append(dict(layer=layer,region=region,family=kind,
                candidate=requests(F,B,4,2)['value'],
                strongest=requests(F,B,8,1)['value']))
    sums={}
    for arm in ('candidate','strongest'):
        keys=('CTA_count','reader_launches','weight_and_halfscale_global_to_shared_requested_bytes',
            'q_and_scale_global_to_shared_requested_bytes','global_scale_global_to_shared_requested_bytes',
            'output_read_plus_write_requested_bytes','owner_write_bytes','executed_warp_MMA_instructions',
            'producer_bulk_copy_instructions','consumer_full_barrier_wait_calls','consumer_empty_barrier_arrival_calls',
            'shared_q_ldmatrix_warp_instructions','shared_scale_fragment_lane_read_bytes')
        sums[arm]={k:sum(f[arm][k] for f in families) for k in keys}
    return dict(rows=B,families=families,totals=sums,
        sum_features=3248640,
        retained_control_contract='All inherited unpaired21/current19/materialized20/originalFW8 and separate/shared controls still execute; these paired standalone totals are additions in the validation harness, not subtractions from it.')


def compose(old):
    result=copy.deepcopy(old['composed_whole_controller'])
    rows=[]
    for p in result['rows']:
        B=p['fresh'];v=copy.deepcopy(p)
        changes=[]
        for layer in range(64):
            for kind in ('attention3' if layer%4==3 else 'input3','ffn2'):
                changes.append(dict(layer=layer,family=kind,**validation_delta(SHAPES[kind],B)))
        sums={k:sum(d[k] for d in changes) for k in (
            'additional_device_requested_bytes','additional_device_allocations',
            'additional_retained_family_host_bytes','additional_Frozen_source_bytes',
            'additional_D2H_bytes','additional_D2H_calls','additional_explicit_fill_bytes',
            'additional_allocation_initial_fill_bytes','additional_reader_launches',
            'added_successful_CUDA_calls_without_release','additional_projection_manifest_groups',
            'validation_added_weight_and_halfscale_requests','validation_added_q_scale_requests')}
        device=sums['additional_device_requested_bytes'];host=sums['additional_retained_family_host_bytes']
        frozen=sums['additional_Frozen_source_bytes'];count=sums['additional_device_allocations']
        for k in ('raw_generation_requests','conditional_all_common_generation_requests','new_family_device_requests'):
            v[k]=p[k]+device
        for k in ('retained_new_family_host_bytes','retained_Stage_and_family_host_bytes','source_successful_postrelease_Stage_and_family_host_bytes'):
            v[k]=p[k]+host
        v['Frozen_source_image_bytes']=p['Frozen_source_image_bytes']+frozen
        v['clone_Frozen_source_copy_bytes']=p['clone_Frozen_source_copy_bytes']+frozen
        v['D2H_projection_bytes']=p['D2H_projection_bytes']+sums['additional_D2H_bytes']
        v['family_device_allocations']=p['family_device_allocations']+count
        v['exact_validation_release_calls_per_full_success']=p['exact_validation_release_calls_per_full_success']+count
        v['ValidationHandle_Arena_retained_logical_bytes']=80*v['exact_validation_release_calls_per_full_success']
        v['Arena_native_pointer_retained_logical_bytes']=8*v['exact_validation_release_calls_per_full_success']
        transient=max(d['additional_preservation_transient_host_bytes'] for d in changes)
        v['source_successful_family_transient_host_upper_bytes']=p['source_successful_family_transient_host_upper_bytes']+transient
        peaks=[]
        for prior in p['early_release_layer_peaks']:
            q=copy.deepcopy(prior)
            local=[d for d in changes if d['layer']==q['layer']]
            extra=sum(d['additional_device_requested_bytes'] for d in local)
            for k in ('current_validation_device_bytes','source_requested_raw_peak','source_requested_common_peak'):
                q[k]=prior[k]+extra
            q['exact_validation_handle_count']=prior['exact_validation_handle_count']+18
            q['minimum_256handle_release_commands']=(q['exact_validation_handle_count']+255)//256
            peaks.append(q)
        v['early_release_layer_peaks']=peaks
        v['source_successful_early_release_generation_raw_peak']=max(q['source_requested_raw_peak'] for q in peaks)
        v['source_successful_early_release_generation_common_peak']=max(q['source_requested_common_peak'] for q in peaks)
        v.update(scope='full64 native22 complete retained-validation source requests; not device execution or physical residency',
            round22_family_deltas=changes,round22_additions=sums,
            paired_candidate_projection_reader_launches=128,
            paired_strongest_projection_reader_launches=128,
            original_primary_reference_launches=128,
            round22_additional_preservation_transient_host_upper_bytes=transient,
            total_all_inherited_plus_paired_family_projection_reader_launches=p['all_five_family_projection_reader_launches']+384,
            field_contract='Unchanged inherited counters remain labelled by their owning round; round22_additions gives explicit new effects. Actual requested-device/host/Frozen/D2H/handle totals and peaks above compose both rounds.')
        rows.append(v)
    configs=[]
    reset=old['inherited_whole_controller']['reset_device_requests_per_stream']
    for p in result['configurations']:
        streams,R=p['streams'],p['fresh'];now,prior=rows[R-1],rows[7];v=copy.deepcopy(p)
        v.update(two_Frozen_source_host_bytes=streams*(prior['Frozen_source_image_bytes']+now['Frozen_source_image_bytes']),
            two_current_generation_Stage_host_bytes=streams*(prior['retained_Stage_and_family_host_bytes']+now['retained_Stage_and_family_host_bytes']),
            source_successful_postrelease_two_Stage_family_host_bytes=streams*(prior['source_successful_postrelease_Stage_and_family_host_bytes']+now['source_successful_postrelease_Stage_and_family_host_bytes']),
            source_successful_current_stage_family_transient_host_upper_bytes=streams*(now['source_successful_current_layer_family_weight_host_upper_bytes']+now['source_successful_family_transient_host_upper_bytes']),
            raw_old8_newR_requested_peak=streams*max(reset+prior['raw_generation_requests'],2*prior['raw_generation_requests'],prior['raw_generation_requests']+now['raw_generation_requests']),
            conditional_all_common_old8_newR_peak=streams*max(reset+prior['conditional_all_common_generation_requests'],2*prior['conditional_all_common_generation_requests'],prior['conditional_all_common_generation_requests']+now['conditional_all_common_generation_requests']),
            source_success_assumed_raw_old8_newR_upper_peak=streams*max(reset+prior['source_successful_early_release_generation_raw_peak'],prior['persistent_Lease_arena_raw_device_bytes']+prior['source_successful_early_release_generation_raw_peak'],prior['persistent_Lease_arena_raw_device_bytes']+now['source_successful_early_release_generation_raw_peak']),
            source_success_assumed_common_old8_newR_upper_peak=streams*max(reset+prior['source_successful_early_release_generation_common_peak'],prior['persistent_Lease_arena_common_device_bytes']+prior['source_successful_early_release_generation_common_peak'],prior['persistent_Lease_arena_common_device_bytes']+now['source_successful_early_release_generation_common_peak']),
            old_new_Arena_validation_handles_retained_logical_bytes=streams*(prior['ValidationHandle_Arena_retained_logical_bytes']+now['ValidationHandle_Arena_retained_logical_bytes']),
            old_new_Arena_native_pointers_retained_logical_bytes=streams*(prior['Arena_native_pointer_retained_logical_bytes']+now['Arena_native_pointer_retained_logical_bytes']))
        configs.append(v)
    result.update(rows=rows,configurations=configs,
        weight_read_bytes_per_stream_segment=result['weight_read_bytes_per_stream_segment']+rows[0]['round22_additions']['validation_added_weight_and_halfscale_requests'],
        weight_read_contract='Inherited21 complete validation streams plus both paired readers and the separately executed unmodifiedFW8 primary-only reference; original controls retained; no cache assumption',
        Frozen='Inherited21 complete image plus both paired arms primary/final/owner0/owner1, original21 direct candidate/control outputs and4B paired launch descriptor per family; adds40*B*sumF+512 per full64 generation.',
        manifest_added_groups_per_layer=8,manifest_added_groups_per_generation=512,
        added_reader_progress_and_manifest_payload=dict(reader_id_bytes=8,logical_added_manifest_id_bytes=512*8,
            ReaderProgress_ABI='unset',allocator_capacity='unset',lookup='all original and new manifest/domain/progress scans retained; data-dependent linear scan work, not timed'))
    result['validation_metadata'].update(retained_handle_count_per_full_generation=14573,
        retained_handle_logical_bytes=14573*80,retained_native_pointer_logical_bytes=14573*8)
    return result


def checks(data):
    assert geometry(4, 1) == dict(ok=False, error='unsupported_paired_geometry')
    assert geometry(4,2)['value']['dynamic_shared_bytes'] == 74112
    assert geometry(4,2)['value']['dynamic_shared_bytes'] == 8*geometry(4,2)['value']['stage_bytes']+128+256
    assert requests(16480, True, 4, 2) == dict(ok=False, error='unsupported_family_or_rows')
    for r in data['one_region_configurations']:
        a, b = r['paired_candidate'], r['equally_strengthened_FW8']
        assert a['CTA_count'] == b['CTA_count']
        for k in ('weight_and_halfscale_global_to_shared_requested_bytes',
                'q_and_scale_global_to_shared_requested_bytes', 'executed_warp_MMA_instructions',
                'output_read_plus_write_requested_bytes', 'reader_launches'):
            assert a[k] == b[k]
        assert a['geometry']['accumulator_payload_bytes_per_CTA'] == b['geometry']['accumulator_payload_bytes_per_CTA']
        assert a['geometry']['accumulator_floats_per_consumer_thread'] == 2*b['geometry']['accumulator_floats_per_consumer_thread']
        old=r['inherited_round21_region']
        assert 2*a['weight_and_halfscale_global_to_shared_requested_bytes'] == old['routes'][0]['weight_scale_requested_bytes']
        assert a['q_and_scale_global_to_shared_requested_bytes'] == old['routes'][4]['q_scale_global_to_shared_requested_bytes']
    assert len(data['one_region_configurations']) == 9
    for row in data['full64_paired_routes']:
        a,b=row['totals']['candidate'],row['totals']['strongest']
        assert a['reader_launches'] == b['reader_launches'] == 128
        assert a['CTA_count'] == b['CTA_count'] == 12720
        assert a['weight_and_halfscale_global_to_shared_requested_bytes'] == b['weight_and_halfscale_global_to_shared_requested_bytes'] == 2339020800
        assert a['q_and_scale_global_to_shared_requested_bytes'] == b['q_and_scale_global_to_shared_requested_bytes'] == 2344550400
        assert 2*a['consumer_empty_barrier_arrival_calls'] == b['consumer_empty_barrier_arrival_calls']
    whole=data['composed_whole_controller']
    assert len(whole['rows'])==8 and len(whole['configurations'])==24
    for row in whole['rows']:
        d=row['round22_additions'];B=row['fresh']
        assert d['additional_device_allocations']==1152
        assert d['additional_device_requested_bytes']==116951040*B
        assert d['additional_retained_family_host_bytes']==129945600*B
        assert d['additional_Frozen_source_bytes']==129945600*B+512
        assert d['additional_D2H_bytes']==168929280*B
        assert d['added_successful_CUDA_calls_without_release']==6016
        assert d['additional_D2H_calls']==1664
        assert row['exact_validation_release_calls_per_full_success']==14573
        assert max(p['exact_validation_handle_count'] for p in row['early_release_layer_peaks'])==248
        assert all(p['minimum_256handle_release_commands']==1 for p in row['early_release_layer_peaks'])


def source_contract():
    required={
        'native/paired.cuh':('FW=4,WG=2,MT=2*WG,NT=4','FW=8,WG=1,MT=2*WG,NT=4',
            'SMEM=RING+GLOBALS','GLOBALS=2*TOKW*4','float acc[2][MT][NT][4]',
            'p.primary[at]=v0','v1+=v0','globals[32+tid]=t<p.rows?p.g[1][t]:0.f'),
        'native/family.cuh':('direct_output=full_output;direct_control=all',
            'allocate_paired(paired_candidate,n)','observe_original_primary(merged_tape,merged_control,original_primary)',
            'full_output=paired_candidate.final_words;control_output=paired_control.final_words',
            'paired_candidate.primary_words,"after_primary_candidate"'),
        'native/native.cu':('append(p->primary_words);append(p->final_words);append(p->owner_words[0]);append(p->owner_words[1])',
            'append(s->direct_output);append(s->direct_control)','merged_reader_launches!=2'),
        'native/binding.hpp':('for(unsigned id=38;id<46;id++)m[l].push_back({id})',),
    }
    for relative,tokens in required.items():
        text=(TOOLS/relative).read_text()
        for token in tokens:
            if token not in text:
                raise ValueError('changed_source_contract:'+relative+':'+token)
    return dict(state='matched',required_dictionaries=len(required),
        contract='Exact source-bound dictionaries plus explicit arithmetic assertions; no CUDA refinement or native-value proof.')


def compiler_cost():
    directory=ROOT/'archive/packing-swarm/round22/compiler1'
    required=('phase-4.log','plan.json','sources.json','resources.log','paired.sass')
    if any(not (directory/name).exists() for name in required):
        return dict(state='unset',reason='complete paired compiler receipts not supplied')
    log=(directory/'phase-4.log').read_text();resources=(directory/'resources.log').read_text()
    sources=json.loads((directory/'sources.json').read_text())
    plan=json.loads((directory/'plan.json').read_text())
    for path in ('native/paired.cuh','native/family.cuh','native/native.cu','native/binding.hpp'):
        p=TOOLS/path
        if sources['sources'].get(str(p))!=sha(p):
            raise ValueError('compiler_source_mismatch:'+path)
    functions={};current=None
    for line in (directory/'paired.sass').read_text().splitlines():
        m=re.search(r'Function : (\S+)',line)
        if m:
            current=m[1];functions[current]=Counter();continue
        m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\b',line)
        if m and current is not None:
            functions[current][m[1]]+=1
    final_path=ROOT/'results/packing-swarm/round22/compiler.json'
    if not final_path.exists():
        raise ValueError('missing_final_compiler_receipt')
    final=json.loads(final_path.read_text())
    if final['state']!='compiled_source_only_zero_GPU_guards' or final['GPU_execution'] is not False or final['source_closure_SHA256']!=plan['source_closure_SHA256']:
        raise ValueError('invalid_final_compiler_receipt')
    arms={}
    for arm,symbol in [('candidate','_ZN8paired229candidateENS_4ArgsE'),('strongest_FW8','_ZN8paired2213strongest_fw8ENS_4ArgsE')]:
        pattern=r'Function properties for '+re.escape(symbol)+r'\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s+: Used (\d+) registers, used (\d+) barriers'
        m=re.search(pattern,log)
        if not m or symbol not in functions:
            raise ValueError('missing_paired_compiler_function:'+symbol)
        stack,stores,loads,registers,barriers=map(int,m.groups())
        resource=re.search(r'Function '+re.escape(symbol)+r':\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+)',resources)
        if not resource:
            raise ValueError('missing_paired_ELF_resource:'+symbol)
        ops=functions[symbol]
        arms[arm]=dict(symbol=symbol,registers_per_thread=registers,stack_bytes_per_thread=stack,
            ptxas_spill_store_bytes=stores,ptxas_spill_load_bytes=loads,ptxas_barriers=barriers,
            ptxas_user_static_shared_bytes=0,ELF_reported_shared_bytes=int(resource[3]),
            dynamic_shared_bytes=74112,SASS_opcode_sites=final['kernels'][symbol]['opcodes'],
            SASS_total_instruction_sites=final['kernels'][symbol]['static_SASS_instructions'],
            raw_disassembly_addressed_line_count=sum(ops.values()),
            raw_disassembly_opcode_counts=dict(sorted(ops.items())),
            raw_count_contract='All address-prefixed instruction lines matching an optional @ or @! predicate and an uppercase opcode; kept separately from the coordinator compiler receipt inventory. These are static parse counts, not executed counts.',
            SASS_local_load_sites=sum(n for op,n in ops.items() if op.split('.')[0]=='LDL'),
            SASS_local_store_sites=sum(n for op,n in ops.items() if op.split('.')[0]=='STL'),
            dynamic_spill_request_bytes='unset',actual_residency='unset',occupancy='unset',native_time='unset')
    return dict(state='set',arms=arms,device_optimization=plan['device_optimization'],
        host_optimization=plan['host_optimization'],GPU_execution=False,
        source_closure_SHA256=plan['source_closure_SHA256'],
        final_receipt=dict(path=str(final_path.relative_to(ROOT)),SHA256=sha(final_path),
            binary_SHA256=final['binary_SHA256'],cubin_SHA256=final['cubin_SHA256'],
            state=final['state'],GPU_execution=False),
        receipts=[dict(path=str((directory/n).relative_to(ROOT)),SHA256=sha(directory/n)) for n in required],
        count_contract='Static SASS instruction sites and ptxas function diagnostics, not dynamic counts/bytes/time. ptxas user static shared and ELF-reported shared are separate fields; no occupancy inferred from either.')


def report():
    contract=source_contract()
    old=json.loads(BASE.read_text())
    configs=[]
    for p in old['one_region_configurations']:
        F, B=p['total_F'],p['batch']
        family_device=p['I']+p['T']+p['W']+16*B*F
        edge=copy.deepcopy(p['supported_edge_standalone_routes'])
        if edge['state']=='logical_source_requests':
            for key in ('candidate_direct','strongest_merged_FW8_direct'):
                route=edge[key]
                oldfamily=route['family_device_bytes']
                route['family_device_bytes']=family_device
                route['combined_device_bytes']+=family_device-oldfamily
            edge['round22_contract']='Paired complete pass/status reader replaces unpaired family source requests in this standalone comparison; original continuation/control ownership and paid preparation/down geometry remain explicit. Materialized20 alternatives above are inherited comparisons, not updated paired routes.'
        configs.append(dict(family=p['family'], rows=B,
            standalone_paired_family=dict(candidate_device_requested_bytes=family_device,
                strongest_device_requested_bytes=family_device,source_and_norm_device_bytes=p['I'],
                one_complete_tape_device_bytes=p['T'],independent_weight_device_bytes=p['W'],
                output_snapshot_twoowners_device_bytes=16*B*F,
                producer_launches=2,reader_launches=1,
                candidate_producer_status_D2H_bytes=p['Z'],strongest_producer_status_D2H_bytes=0,
                producer_status_contract='Existing owned candidate prep observes its two producer status arrays; strongest literal prep does not. Both Tape layouts allocate and fill those arrays; no allocation is subtracted in the matched standalone family geometry.'),
            standalone_supported_edge_routes=edge,
            inherited_round21_region=p,
            paired_candidate=requests(F,B,4,2)['value'],
            equally_strengthened_FW8=requests(F,B,8,1)['value'],
            round22_validation_delta=validation_delta(F,B)))
    return dict(state='static_exact_source_request_arithmetic', CPU_only=True,
        source_contract=contract,compiler_cost=compiler_cost(),
        source=[dict(path=str(p.relative_to(ROOT)),SHA256=sha(p)) for p in
            [TOOLS/'cost/invoice.py',TOOLS/'native/generate.py',TOOLS/'native/paired.cuh',
             TOOLS/'native/family.cuh',TOOLS/'native/native.cu',TOOLS/'native/binding.hpp',
             TOOLS/'native/generation.hpp',TOOLS/'native/session.cuh',TOOLS/'native/stage.cuh',
             TOOLS/'native/views.cuh',TOOLS/'source.hpp']],
        GPU_execution=False, nomination='none', new_packing_bound=False,
        inherited_source=dict(path=str(BASE.relative_to(ROOT)), SHA256=sha(BASE),
            contract='All unchanged Round21 costs are retained by exact reference. Current deltas and comparison are explicit; original controls are not silently deleted.'),
        one_region_configurations=configs,
        full64_paired_routes=[full64_paired_routes(B) for B in range(1,9)],
        composed_whole_controller=compose(old),
        E_D=old['E_D'], inherited_whole_controller=old['inherited_whole_controller'],
        inherited_receiver_capacity=old['inherited_receiver_capacity'],
        original_A8_INT8_endpoint_contract='Unchanged distinct endpoints, not FP4 control substitutes; per-region full pricing retained in inherited_round21_region.original_A8_controls.',
        custody_cost=old['custody_cost'], inherited_native_ABI=old['native_abi'],
        physical_unsets=old['unsets'],
        paired_geometry_comparison=dict(candidate=geometry(4,2)['value'],
            equally_strengthened_FW8=geometry(8,1)['value']),
        inherited_cost_scope='Includes all source/norm factories and strides; E/D, half/global scales, quantizers and padded64 tapes; original19, materialized20 and direct21 continuations; ring/convpending/bothpending/checkpoint/F16KV/head/logits/argmax; source preservation, status/fault owners and reset; scans, metadata, borrow views, Frozen/old-new/deepclone and D2D/before-after images; retention on uncertain effects, request/evidence storage, serialization/rename/directoryfsync and whole release. None is a zero charge.',
        conclusion='Both arms halve recurring weight/halfscale requests and reader launches against their unpaired origins. Equal tile staging removes the R4 q/scale duplication. Shared and aggregate accumulator payload match between paired arms; candidate doubles accumulator payload per consumer thread. Bytes and reader launches match against the equally strengthened control; fewer consumer barrier/shared-q/scale instructions are bought with doubled per-thread accumulator liveness. Compiled candidate spills while control does not. No nominee, time/residency/occupancy claim.')


def markdown(data):
    lines=['# Round22 — equal-boundary paired-reader invoice','',
        'CPU-only exact source requests. No GPU execution, spending or physical-time claim.','',
        '## Paired geometry','',
        'Both arms: eight32-feature groups/256features per CTA, TOKW32, K5120, forty ascendingK128 folds per independent pass. Each stage is4096W+512halfscales+2048q0+256s0+2048q1+256s1=9216B; eight stages plus sixteen8B mbarriers and256B shared global scales request74112B dynamic shared. Both arms load the two actual global-scale vectors once per CTA (8*rows*CTAs bytes), not per warp or output lane.', '',
        'Candidate: four consumer warps/WG2 plus one producer (160threads),128 independent accumulator floats per consumer thread. Strongest: eight consumer warps/WG1 plus one producer (288threads),64 floats/thread. Both retain65536B aggregate accumulator payload/CTA. This is source liveness, not ptxas registers or occupancy.','',
        '| Family | Rows | CTAs | Paired W+halfscale B | Paired q+s B (each arm) | Each arm launches |',
        '|---|---:|---:|---:|---:|---:|']
    for r in data['one_region_configurations']:
        a=r['paired_candidate']
        lines.append(f"| {r['family']} | {r['rows']} | {a['CTA_count']} | {a['weight_and_halfscale_global_to_shared_requested_bytes']} | {a['q_and_scale_global_to_shared_requested_bytes']} | 1 |")
    lines += ['',data['conclusion'],'',
        '## Complete inherited invoice','',
        data['inherited_cost_scope'],'',
        'The source-bound JSON retains every regional and full64 inherited cost. Standalone source requests are compared separately from the complete retained validation harness. Both paired arms use identical source+Norm, one full Tape (physical64-row q/s padding, globals, transformed/Norm and producer statuses), one independent prepared weight image and16BF output/snapshot/two-owner bytes. Original candidate owned prep still pays its producer status D2H/checks; generic strongest prep does not. Direct preparation/down continuation charges remain exactly inherited, including the candidate co-reader and strongest generic control.','',
        'Both new arms are appended, not substituted for old controls. Each family adds four FP32 images, four owner images and a separately allocated/launched originalFW8 primary reference: nine allocations/36BF device bytes. Initial fills36BF plus explicit zeros/owner poison28BF; complete new observations/original primary/end-preservation add52BF D2H (13calls). New host retention40BF, original-primary transient4BF plus prolonged originalFW8 final-vector transient4BF, preservation transient16BF, Frozen40BF+4B. Both arms have complete primary/final and owner observations. Original/direct outputs and new primary snapshots survive to preservation, and uncertain backing stays owned.','',
        'Per family successful added CUDA calls before release:9allocations+9initialfills+7explicitfills+3attributes+3launches+3fences+13D2H=47. Four new pass manifest groups add12 start/launched/observed calls. Native full64 adds512 groups (38..45/layer); all complete manifest/progress/domain/evidence scans remain paid. ReaderProgress/control-block/vector/allocator capacities remain unset, not zero.','',
        'Full64 each paired arm requests2339020800B weight/halfscales,2344550400B q/scales,101760*rowsB globals,128reader launches and12720CTAs. Original independent/retained controls still execute: validation adds384launches including128primary references,7017062400B weight/halfscale and5861376000B q/scale requests; no cache reduction is assumed.','',
        '| Fresh rows | Retained-all generation B | Success-assumed layer-local peak B | Added device B | Added D2H B |',
        '|---|---:|---:|---:|---:|']
    for r in data['composed_whole_controller']['rows']:
        if r['fresh'] in (1,8):
            d=r['round22_additions']
            lines.append(f"| {r['fresh']} | {r['raw_generation_requests']} | {r['source_successful_early_release_generation_raw_peak']} | {d['additional_device_requested_bytes']} | {d['additional_D2H_bytes']} |")
    lines += ['',
        'Full64 adds1152 whole-array validation handles:14573 total (80B records plus8B pointer entries), at most248/layer under256. Host/Frozen deltas are129945600*rows and129945600*rows+512B. Old8/newR, reset, deepclone copies and enlarged host images are recomposed for all24 serialized independent-stream configurations; none is physical residency.','']
    compiler=data['compiler_cost']
    if compiler['state']=='set':
        a,b=compiler['arms']['candidate'],compiler['arms']['strongest_FW8']
        lines += ['## Compiled liveness loss','',
            f"sm120a candidate:{a['registers_per_thread']}registers/thread,{a['stack_bytes_per_thread']}B stack,{a['ptxas_spill_store_bytes']}B ptxas spill stores/{a['ptxas_spill_load_bytes']}B spill loads. Strongest:{b['registers_per_thread']}registers/thread,zero stack/spills. Both request74112B dynamic shared; ptxas reports no user static shared, ELF resource shared1024B is separately retained. Actual host:{compiler['host_optimization']}; device:{compiler['device_optimization']}.", '',
            'SASS opcode/site counts are bound in JSON, including actual local spill sites. Dynamic spill bytes, barrier-spin iterations, residency and occupancy are unset. Source barrier arrivals/waits and shared q/scale fragment instructions are lower for four consumer warps; the matched bytes/launches and compiled spill regression prevent nomination. This is not an absolute Pareto claim across every instruction metric.','']
    lines += [
        '**Unset:** native words/faults/time, concreteCUDArefinement, allocator capacity/granularity/residency, cache-effective traffic, occupancy, tensor-wideB8/B32 controller, engine/model/task/HTTP.','',
        'Replay: `python3 method/packing-swarm/tools/round22/cost/invoice.py --check`; regenerate with no flag.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group()
    m.add_argument('--check',action='store_true')
    m.add_argument('--verify',action='store_true')
    a=p.parse_args()
    try:
        data=report();checks(data)
        if a.verify and (json.loads((OUT/'invoice.json').read_text()) != data or (OUT/'COST.md').read_text() != markdown(data)):
            print(json.dumps(dict(ok=False,error='saved_source_or_arithmetic_mismatch')));return 2
        if not a.check and not a.verify:
            OUT.mkdir(parents=True,exist_ok=True)
            (OUT/'invoice.json').write_text(json.dumps(data,indent=2)+'\n')
            (OUT/'COST.md').write_text(markdown(data))
    except (OSError,ValueError,KeyError,TypeError) as e:
        print(json.dumps(dict(ok=False,error=type(e).__name__,detail=str(e))));return 2
    print(json.dumps(dict(ok=True,state=data['state'],regions=9,GPU_execution=False,nomination=data['nomination'])));return 0


if __name__=='__main__':
    raise SystemExit(main())
