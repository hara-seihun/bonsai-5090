#!/usr/bin/env python3
import collections
import pathlib
import re
import struct
import sys

root = pathlib.Path(sys.argv[1])
with open(sys.argv[2], 'rb') as f:
    def unpack(fmt):
        size = struct.calcsize(fmt)
        value = f.read(size)
        if len(value) != size:
            raise ValueError('truncated GGUF')
        return struct.unpack(fmt, value)[0]

    def string():
        size = unpack('<Q')
        value = f.read(size)
        if len(value) != size:
            raise ValueError('truncated GGUF string')
        return value.decode()

    def skip_value(kind):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind in sizes:
            f.seek(sizes[kind], 1)
        elif kind == 8:
            string()
        elif kind == 9:
            subtype, count = unpack('<I'), unpack('<Q')
            for _ in range(count):
                skip_value(subtype)
        else:
            raise ValueError(f'unknown metadata type {kind}')

    if f.read(4) != b'GGUF' or unpack('<I') != 3:
        raise ValueError('expected GGUF v3')
    tensor_count, metadata_count = unpack('<Q'), unpack('<Q')
    for _ in range(metadata_count):
        string()
        skip_value(unpack('<I'))
    types = collections.Counter()
    for _ in range(tensor_count):
        string()
        dimensions = unpack('<I')
        f.seek(8 * dimensions, 1)
        types[unpack('<I')] += 1
        unpack('<Q')

header = (root / 'ggml/include/ggml.h').read_text()
names = {int(number): name for name, number in re.findall(r'GGML_TYPE_(\w+)\s*=\s*(\d+)', header)}
required = {names[kind] for kind in types}
print('MODEL_TENSOR_TYPES', {names[kind]: count for kind, count in types.items()}, flush=True)
(root.parent / 'tensor-types.txt').write_text(str({names[kind]: count for kind, count in types.items()}) + '\n')
cuda = root / 'ggml/src/ggml-cuda'
mmq = cuda / 'mmq.cu'
text = mmq.read_text()
start, end = text.index('static void ggml_cuda_mul_mat_q_switch_type'), text.index('static size_t ggml_cuda_mmq_q8_buffer_size')
body = text[start:end]
pattern = r'        case GGML_TYPE_(\w+):\n            mul_mat_q_case<GGML_TYPE_\1>\(ctx, args, stream\);\n            break;\n'
body = re.sub(pattern, lambda match: match[0] if match[1] in required else '', body)
mmq.write_text(text[:start] + body + text[end:])
cmake = cuda / 'CMakeLists.txt'
text = cmake.read_text()
needle = '    ggml_add_backend_library(ggml-cuda\n'
if text.count(needle) != 1:
    raise ValueError('unexpected CUDA CMake structure')
keep = '|'.join(name.lower() for name in sorted(required))
focus = f'''    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "/fattn(-tile)?\\\\.cu$|/template-instances/fattn")
    file(GLOB FOCUSED_MMQ "template-instances/mmq*.cu")
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "/template-instances/mmq")
    list(FILTER FOCUSED_MMQ INCLUDE REGEX "mmq-instance-({keep})\\\\.cu$")
    list(APPEND GGML_SOURCES_CUDA ${{FOCUSED_MMQ}})
    list(APPEND GGML_SOURCES_CUDA "${{CMAKE_CURRENT_SOURCE_DIR}}/reference-no-fa.cu")
'''
cmake.write_text(text.replace(needle, focus + needle))
(cuda / 'reference-no-fa.cu').write_text('''#include "fattn.cuh"
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("Flash attention is disabled in the focused reference build");
}
bool ggml_cuda_flash_attn_ext_supported(int, const ggml_tensor *) { return false; }
size_t ggml_cuda_flash_attn_ext_get_alloc_size(int, const ggml_tensor *) {
    GGML_ABORT("Flash attention is disabled in the focused reference build");
}
''')
print('Focused MMQ types:', sorted(required), flush=True)
