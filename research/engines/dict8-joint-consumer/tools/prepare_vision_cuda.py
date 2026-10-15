#!/usr/bin/env python3
"""Materialize a pinned encoder-only Prism source tree; leave the upstream tree untouched."""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import tarfile

REVISION = '1a07bfa5f4144274c8f1c9963821dd9d9a51854b'
p = argparse.ArgumentParser()
p.add_argument('--upstream', required=True)
p.add_argument('--source', required=True)
p.add_argument('--build', required=True)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--cpu-build', help='memoized same-revision CPU llama/mtmd libraries; compile only ggml CUDA backend')
p.add_argument('--compile', action='store_true')
a = p.parse_args()
upstream, source, build = map(lambda x: pathlib.Path(x).resolve(), (a.upstream, a.source, a.build))
if source == upstream or upstream in source.parents or source in upstream.parents:
    raise SystemExit('focused source must be separate from the upstream tree')
stub = pathlib.Path(__file__).with_name('vision-cublas.cu')
signature = hashlib.sha256(pathlib.Path(__file__).read_bytes() + stub.read_bytes()).hexdigest()
manifest = source / 'halo-vision-build.json'
expected = {'revision': REVISION, 'recipe_sha256': signature, 'types': ['F32', 'F16', 'BF16', 'Q8_0'], 'matmul': 'cuBLAS', 'flash_attention': False}

def body_span(text, name):
    at = text.index(name)
    start = text.index('{', at)
    # Skip comments and literals while matching C++ braces.
    tokens = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|[{}]', re.S)
    depth = 0
    for token in tokens.finditer(text, start):
        if token.group() == '{': depth += 1
        elif token.group() == '}':
            depth -= 1
            if depth == 0: return start, token.end()
    raise ValueError('unbalanced function: '+name)

def replace_body(text, name, body):
    start, end = body_span(text, name)
    return text[:start] + '{\n'+body+'\n}' + text[end:]

if not manifest.exists() or json.loads(manifest.read_text()) != expected:
    source.mkdir(parents=True, exist_ok=True)
    if not manifest.exists() or json.loads(manifest.read_text())['revision'] != REVISION:
        archive = subprocess.Popen(['git','-C',str(upstream),'archive',REVISION], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=archive.stdout, mode='r|') as files:
            files.extractall(source, filter='data')
        if archive.wait(): raise SystemExit('cannot extract pinned Prism source')
    cuda = source / 'ggml/src/ggml-cuda'
    cmake = cuda / 'CMakeLists.txt'
    text = subprocess.check_output(['git','-C',str(upstream),'show',REVISION+':ggml/src/ggml-cuda/CMakeLists.txt'], text=True)
    anchor = '    ggml_add_backend_library(ggml-cuda\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    # This tree owns only the Q8/F16/F32 vision encoder, not a language backend.
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "template-instances/")
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "/(fattn[^/]*|mmq|mmvq|mmf|mmvf)\\\\.cu$")
    list(APPEND GGML_SOURCES_CUDA "${CMAKE_CURRENT_SOURCE_DIR}/vision-cublas.cu")
    list(REMOVE_DUPLICATES GGML_SOURCES_CUDA)

'''+anchor)
    if cmake.read_text() != text: cmake.write_text(text)
    if not (cuda/'vision-cublas.cu').exists() or (cuda/'vision-cublas.cu').read_bytes() != stub.read_bytes():
        (cuda/'vision-cublas.cu').write_bytes(stub.read_bytes())
    dispatch = cuda/'ggml-cuda.cu'
    text = subprocess.check_output(['git','-C',str(upstream),'show',REVISION+':ggml/src/ggml-cuda/ggml-cuda.cu'], text=True)
    text = replace_body(text, 'static void ggml_cuda_mul_mat(', '''    switch (src0->type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16: case GGML_TYPE_Q8_0:
            ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
            return;
        default:
            GGML_ABORT("encoder-only CUDA backend: unsupported matmul weight type %s", ggml_type_name(src0->type));
    }''')
    text = replace_body(text, 'static void ggml_cuda_mul_mat_id(', '    GGML_ABORT("encoder-only CUDA backend: MoE matmul is unsupported");')
    for name in ('static bool ggml_cuda_should_fuse_mul_mat(', 'static bool ggml_cuda_should_fuse_mul_mat_vec_f(', 'static bool ggml_cuda_should_fuse_mul_mat_vec_q('):
        text = replace_body(text, name, '    return false;')
    start, end = body_span(text, 'static bool ggml_backend_cuda_device_supports_op(')
    text = text[:start+1] + '''
    if (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_FLASH_ATTN_EXT) return false;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && ggml_is_quantized(op->src[i]->type) && op->src[i]->type != GGML_TYPE_Q8_0) return false;
    }
''' + text[start+1:]
    if dispatch.read_text() != text: dispatch.write_text(text)
    manifest.write_text(json.dumps(expected, indent=2)+'\n')
command = ['cmake','-S',str(source),'-B',str(build),'-DGGML_CUDA=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=120a-real','-DGGML_CUDA_FA=OFF','-DGGML_CUDA_FORCE_CUBLAS=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=ON','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_CURL=OFF']
subprocess.run(command, check=True)
# Check the owning build graph, not just the CMake option.
makefile = (build/'ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/build.make').read_text()
assert 'template-instances/' not in makefile
assert not re.search(r'/ggml-cuda/(?:mmq|mmvq|mmf|mmvf|fattn[^/]*)\.cu', makefile)
target = 'ggml' if a.cpu_build else 'mtmd'
print(json.dumps({'source':str(source),'build':str(build),'profile':expected,'build_command':['cmake','--build',str(build),'--target',target,'-j'+str(a.threads)]}, indent=2))
if a.compile:
    subprocess.run(['cmake','--build',str(build),'--target',target,'-j'+str(a.threads)], check=True)
    if a.cpu_build:
        revision = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()
        if revision != REVISION: raise SystemExit('memoized CPU libraries require the pinned upstream HEAD')
        cpu_bin = pathlib.Path(a.cpu_build).resolve()/'bin'
        reused = {}
        # mtmd has no CUDA-specific compilation; it dispatches via the ggml backend registry.
        # Keep the focused CUDA ggml libraries; only reuse the unchanged public bridge/language ABI.
        for prefix in ('libllama.so', 'libmtmd.so'):
            entries = list(cpu_bin.glob(prefix+'*'))
            if not entries: raise SystemExit('missing memoized library: '+str(cpu_bin/prefix))
            for entry in entries:
                dst = build/'bin'/entry.name
                if dst.is_symlink(): dst.unlink()
                shutil.copy2(entry, dst)
                reused[entry.name] = hashlib.sha256(dst.read_bytes()).hexdigest()
        (build/'halo-reused-libraries.json').write_text(json.dumps({'upstream_revision':revision,'cpu_build':str(cpu_bin.parent),'sha256':reused}, indent=2)+'\n')
