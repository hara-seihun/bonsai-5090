#!/usr/bin/env python3
"""Emit only the registered integrated compilation contract, or a typed error."""
import json
from pathlib import Path
import sys

REQUIRED = {'H5_PROFILE_GEMM', 'H5_PDL_EARLY_GEMM_RELEASE', 'H5_PDL_GDN_STATE_OVERLAP',
            'H5_NO_L2PF', 'H5_PROFILE_SCALE_CARRIER', 'H5_WEIGHT_TMA_CACHE_POLICY=2'}


def flags(path):
    try:
        spec = json.loads(path.read_text())
        values = spec['build_defines']
        if spec['schema'] != 'halo-integrated-serving-v1' or not isinstance(values, list) or not all(type(x) is str for x in values):
            return {'state': 'error', 'code': 'invalid_build_contract'}
        if len(values) != len(REQUIRED) or set(values) != REQUIRED:
            return {'state': 'error', 'code': 'incomplete_or_unknown_build_contract'}
        return {'state': 'ready', 'flags': ' '.join('-D' + value for value in values)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'state': 'error', 'code': 'build_contract_input', 'detail': str(error)}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: selected_build_defines.py PRESET_JSON', file=sys.stderr); sys.exit(2)
    result = flags(Path(sys.argv[1]))
    if result['state'] == 'error':
        print(json.dumps(result), file=sys.stderr); sys.exit(1)
    print(result['flags'])
