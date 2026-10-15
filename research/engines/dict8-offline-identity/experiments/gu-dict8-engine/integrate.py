from pathlib import Path
import hashlib,json
p=Path(__file__).parent;src=p.parents[1]/'src'
s=(p/'frozen-engine.cu').read_text()
assert hashlib.sha256((p/'frozen-engine.cu').read_bytes()).hexdigest()=='e2ae33c409fc8a4fe668f8c5cbf51009f65ea76745741fd5d3135c329a5b18c4'
def edit(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
edit('#include "deterministic_gemm.cuh"','#include "deterministic_gemm.cuh"\n#include "dict8.cuh"\n#include "dict8_codec.hpp"\n#include "dict8_cache.hpp"\n#include "dict8_model.hpp"')
edit('namespace h5 {', '\n'.join(['namespace h5 {',
 'static thread_local int dict8_cache_error=0;',
 'static void cache_require(dict8::cache::Error e){dict8_cache_error=int(e);throw std::runtime_error(std::string("dict8-cache:")+dict8::cache::name(e));}',
 'template<class T> static T cache_take(dict8::cache::Result<T> r){if(auto* e=std::get_if<dict8::cache::Error>(&r))cache_require(*e);return std::move(std::get<T>(r));}']))
edit('    g.open(path);', '    if(dict8_enabled){dict8_model=cache_take(dict8::cache::open_model(path));dict8_preparse_ns=dict8_model->costs().identity_ns;dict8_cache=cache_take(dict8::cache::Reader::open(getenv("H5_GU_DICT8_CACHE"),*dict8_model,dict8_model->identity()));}else g.open(path);')
edit('  Gguf g;', '  Gguf g;\n  std::unique_ptr<dict8::cache::ExactModel> dict8_model;\n  const Gguf& model()const{return dict8_model?dict8_model->gguf():g;}')
# Tensor/metadata consumers all select the same owned model. The unset opener stays original.
s=s.replace('g.t(', 'model().t(').replace('g.kv', 'model().kv')
edit('    if (const char* trace = getenv("H5_SAMPLE_TRACE_FILE")) {','    dict8_cache_error=0;\n    if (const char* trace = getenv("H5_SAMPLE_TRACE_FILE")) {')
edit('bool drafter_requant = false;', 'bool drafter_requant = false; uint8_t* dict8_weights=nullptr;')
edit('bool scale_carrier = false, carrier_wf0 = false;', '''bool scale_carrier = false, carrier_wf0 = false;
  bool dict8_enabled=false,dict8_check=false;
  uint32_t* dict8_book=nullptr;uint32_t* dict8_stats=nullptr;
  float *dict8_pairs=nullptr,*dict8_ref_pairs=nullptr,*dict8_ref=nullptr,*dict8_gref=nullptr;
  uint8_t *dict8_qref=nullptr,*dict8_sref=nullptr;
  uint64_t dict8_setup[10]{};
  std::unique_ptr<dict8::cache::Reader> dict8_cache;
  uint64_t dict8_preparse_ns=0,dict8_repack_ns=0,dict8_original_upload_ns=0,dict8_means_ns=0;''')
edit('bool interleave = false) {\n    std::vector<const GTensor*> ts;', 'bool interleave = false, bool dict8_prepare=false) {\n    std::vector<const GTensor*> ts;')
edit('return tmat(ts, K, role, interleave);', 'return tmat(ts, K, role, interleave, dict8_prepare);')
edit('bool interleave = false) {\n    int N = 0;', 'bool interleave = false, bool dict8_prepare=false) {\n    int N = 0;')
a='''      } else { m.w4 = upload<uint4>(W.data(), W.size() / 16); m.s4 = upload<__half>(S.data(), S.size()); m.wf = compact_weights ? 3 : 0; }
'''
b=a+'''      if(dict8_prepare) {
        if(!dict8_enabled||!wf0||m.wf!=0||!interleave||K!=D||N!=GU||ts.size()!=2||W.size()!=size_t(GU/32)*(D/128)*1024)
          throw std::runtime_error("dict8 original WF0 GU preparation contract");
        const auto index=uint32_t(dict8_setup[1]);
        if(ts[0]->name!="blk."+std::to_string(index)+".ffn_gate.weight"||ts[1]->name!="blk."+std::to_string(index)+".ffn_up.weight")
          cache_require(dict8::cache::Error::ModelLayout);
        auto image=cache_take(dict8_cache->layer(index));
        const auto encoded=std::chrono::steady_clock::now();
        m.dict8_weights=upload<uint8_t>(image.data(),image.size());
        const auto uploaded=std::chrono::steady_clock::now();
        dict8_setup[1]++;dict8_setup[2]+=image.size();dict8_setup[4]+=W.size();dict8_setup[5]+=S.size()*2;
        dict8_setup[9]+=std::chrono::duration_cast<std::chrono::nanoseconds>(uploaded-encoded).count();
      }
'''
edit(a,b)
edit('    if (use_fp4) {\n      std::vector<uint8_t> W', '    if (use_fp4) {\n      const auto dict8_original_begin=dict8_prepare?dict8::cache::tick():0;\n      uint64_t dict8_original_ready=0;\n      std::vector<uint8_t> W')
edit('      m.il = interleave;','      if(dict8_prepare){dict8_original_ready=dict8::cache::tick();dict8_repack_ns+=dict8_original_ready-dict8_original_begin;}\n      m.il = interleave;')
edit('      if(dict8_prepare) {','      if(dict8_prepare) {\n        dict8_original_upload_ns+=dict8::cache::tick()-dict8_original_ready;')
edit('      m.mean_scale = dalloc<float>(m.N);','      const auto dict8_mean_begin=dict8_prepare?dict8::cache::tick():0;\n      m.mean_scale = dalloc<float>(m.N);')
edit('      weight_bytes += (size_t)m.N * sizeof(float);','      if(dict8_prepare)dict8_means_ns+=dict8::cache::tick()-dict8_mean_begin;\n      weight_bytes += (size_t)m.N * sizeof(float);')
edit('carrier_wf0 = experiment("H5_CARRIER_WF0");','''carrier_wf0 = experiment("H5_CARRIER_WF0");
    dict8_enabled=experiment("H5_GU_DICT8");dict8_check=experiment("H5_GU_DICT8_CHECK");
    if(dict8_check&&!dict8_enabled)throw std::runtime_error("dict8 checks require dict8 candidate");
    if(dict8_enabled&&(!carrier_wf0||!scale_carrier||!compact||!act_fp4||!use_fp4||wscale_e4m3||
       getenv("H5_NO_GUFUSE")||max_rows<64||fuse_gu_selected()||
       (getenv("H5_MEGA")&&atoi(getenv("H5_MEGA")))))
      throw std::runtime_error("dict8 requires original interleaved WF0 scale-carrier graph route");
    if(dict8_enabled) {
      for(const char* name:{"H5_TAIL_PREP","H5_GEMM_FLAG","H5_GU_SPLIT","H5_GU_PLAIN_ROWS","H5_ORDERED_PREP","H5_GU_COOP"})
        if(getenv(name))throw std::runtime_error(std::string("dict8 excludes alternate GU schedules: ")+name);
      if((getenv("H5_F02")&&strcmp(getenv("H5_F02"),"0"))||(getenv("H5_F07")&&strcmp(getenv("H5_F07"),"0")))
        throw std::runtime_error("dict8 excludes F02/F07 routes");
      if((getenv("H5_GEMM_PREWAIT")&&strcmp(getenv("H5_GEMM_PREWAIT"),"5"))||
         (getenv("H5_GEMM_L2PF")&&strcmp(getenv("H5_GEMM_L2PF"),"0")))
        throw std::runtime_error("dict8 requires matched prewait5 and L2PF0");
    }''')
edit('    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));', '''    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    if(dict8_enabled) {
      const auto book=dict8::dictionary();dict8_book=upload<uint32_t>(book.data(),book.size());
      dict8_setup[0]=1;dict8_setup[3]=book.size()*4;
    }''')
edit('y.gu = tmat({p + "ffn_gate.weight", p + "ffn_up.weight"}, D, TargetProjection::Block, !getenv("H5_NO_GUFUSE"));', 'y.gu = tmat({p + "ffn_gate.weight", p + "ffn_up.weight"}, D, TargetProjection::Block, !getenv("H5_NO_GUFUSE"),dict8_enabled);')
edit('    fprintf(stderr, "\\nweights %.2f GB in %.1f s\\n", weight_bytes / 1e9, now_s() - t0);','''    if(dict8_enabled) {
      cache_take(dict8_cache->finish());
      if(dict8_setup[1]!=NLAYER)throw std::runtime_error("dict8 incomplete layer preparation");
      dict8_setup[6]=size_t(NLAYER)*GU*sizeof(float);dict8_setup[7]=weight_bytes;
    }
    fprintf(stderr, "\\nweights %.2f GB in %.1f s\\n", weight_bytes / 1e9, now_s() - t0);''')
edit('    q4_alt = dalloc<uint8_t>((size_t)64 * FF / 2);','''    if(dict8_check) {
      dict8_stats=dalloc<uint32_t>(12);dict8_pairs=dalloc<float>(size_t(64)*GU);dict8_ref_pairs=dalloc<float>(size_t(64)*GU);
      dict8_ref=dalloc<float>(size_t(64)*FF);dict8_qref=dalloc<uint8_t>(size_t(64)*FF/2);
      dict8_sref=dalloc<uint8_t>(size_t(64)*FF/16);dict8_gref=dalloc<float>(64);
    }
    q4_alt = dalloc<uint8_t>((size_t)64 * FF / 2);''')
edit('''    const uint8_t* A = q4; const uint8_t* AS = s4; const float* AG = g4;
    const auto f07_route = select_f07_route''','''    const uint8_t* A = q4; const uint8_t* AS = s4; const float* AG = g4;
    if(dict8_enabled&&R==64&&m.dict8_weights) {
      auto key=std::make_tuple(m.N,m.K,rbucket(R),false,101);auto it=tuned.find(key);
      if(it==tuned.end()||it->second.first!=13||it->second.second!=1||m.wf!=0||!m.il||!m.mean_scale)
        throw std::runtime_error("dict8 requires matched frozen GU cfg13/split1");
      CK(launch_gu_dict8(A,AS,AG,R,m.K,m.dict8_weights,m.N,out,ldo,m.mean_scale,dict8_book,st,dict8_check?dict8_pairs:nullptr));
      if(dict8_check) {
        CK(launch_gu_dict8_reference(A,AS,AG,R,m.K,reinterpret_cast<const uint8_t*>(m.w4),m.N,dict8_ref,ldo,m.mean_scale,st,dict8_ref_pairs));
        CK(compare_dict8_pairs(dict8_pairs,dict8_ref_pairs,out,dict8_ref,dict8_stats,st));
      }
      return;
    }
    const auto f07_route = select_f07_route''')
# Insert checks only at the ordinary GU observer seam; the original consumer remains intact.
a='''        prep(proj, nullptr, FF, nullptr, signs17408, FF, P_SIGN | P_HADAMARD | P_QUANT, R);
      } else {
        gemm(Ly.gu, R, proj, GU, false);'''
b='''        if(dict8_check&&R==64) {
          auto aq=q4,as=s4;auto ag=g4;q4=dict8_qref;s4=dict8_sref;g4=dict8_gref;
          prep(dict8_ref,nullptr,FF,nullptr,signs17408,FF,P_SIGN|P_HADAMARD|P_QUANT,R);
          q4=aq;s4=as;g4=ag;
        }
        prep(proj, nullptr, FF, nullptr, signs17408, FF, P_SIGN | P_HADAMARD | P_QUANT, R);
        if(dict8_check&&R==64)CK(compare_dict8_quant(q4,dict8_qref,s4,dict8_sref,g4,dict8_gref,dict8_stats,st));
      } else {
        gemm(Ly.gu, R, proj, GU, false);'''
edit(a,b)
s+='''
extern "C" __declspec(dllexport) int h5_dict8_setup_stats(void* e,uint64_t* out,int count){
 if(!e||!out||count!=10)return -1;auto* en=static_cast<h5::Engine*>(e);
 std::memcpy(out,en->dict8_setup,10*sizeof(uint64_t));return 0;
}
extern "C" __declspec(dllexport) int h5_dict8_check_stats(void* e,uint32_t* out,int count){
 if(!e||!out||count!=12)return -1;auto* en=static_cast<h5::Engine*>(e);if(!en->dict8_stats)return -2;
 return int(cudaMemcpy(out,en->dict8_stats,12*sizeof(uint32_t),cudaMemcpyDeviceToHost));
}
extern "C" __declspec(dllexport) int h5_dict8_reset_checks(void* e){
 if(!e)return -1;auto* en=static_cast<h5::Engine*>(e);if(!en->dict8_stats)return -2;
 auto error=cudaMemsetAsync(en->dict8_stats,0,12*sizeof(uint32_t),en->st);
 if(error!=cudaSuccess)return int(error);return int(cudaStreamSynchronize(en->st));
}
'''
s+='''
extern "C" __declspec(dllexport) int h5_dict8_cache_last_error(){return h5::dict8_cache_error;}
extern "C" __declspec(dllexport) int h5_dict8_cache_stats(void* e,uint64_t* out,int count){
 if(!e||!out||count!=13)return -1;auto* en=static_cast<h5::Engine*>(e);if(!en->dict8_cache)return -2;
 auto c=en->dict8_cache->costs;uint64_t v[]={c.model_bytes,c.model_hash_ns,c.open_header_ns,c.read_bytes,c.read_ns,c.hash_ns,c.checked_layers,c.allocation_ns,c.finish_ns,en->dict8_preparse_ns,en->dict8_repack_ns,en->dict8_original_upload_ns,en->dict8_means_ns};std::memcpy(out,v,sizeof v);return 0;
}
'''
s+='''
extern "C" DICT8_EXPORT int h5_dict8_identity_stats(void* e,uint64_t* out,int count){
 if(!e||!out||count!=14)return -1;auto* en=static_cast<h5::Engine*>(e);if(!en->dict8_model)return -2;
 const auto& c=en->dict8_model->costs();uint64_t v[]={c.source_open_ns,c.allocation_ns,c.read_copy_ns,c.protection_ns,c.identity_ns,c.parse_ns,c.source_read_bytes,c.copied_bytes,c.allocated_bytes,c.identity_bytes,c.identity_scans,c.mapped_bytes,c.admission_ns,c.storage};std::memcpy(out,v,sizeof v);return 0;
}
'''
for create,args in [('h5_create','path, slots, ctx, rows'),('h5_create_draft','path, slots, ctx, rows, proposals')]:
 old='try { return new Engine('+args+'); } catch (std::exception& e) {'
 new='h5::dict8_cache_error=0;try { return new Engine('+args+'); } catch (const std::bad_alloc&) { h5::dict8_cache_error=int(h5::dict8::cache::Error::Allocation); return nullptr; } catch (std::exception& e) { if(!h5::dict8_cache_error)h5::dict8_cache_error=int(h5::dict8::cache::Error::Initialization);'
 edit(old,new)
 signature='void* '+create+'(const char* path, int slots, int ctx, int rows'+(', int proposals' if create.endswith('draft') else '')+') {'
 edit(signature,signature+'\n  h5::dict8_cache_error=0;')
edit('  const auto cfg = h5::draft_config(proposals, slots, false, ctx, 1);', '  h5::dict8_cache_error=int(h5::dict8::cache::Error::Initialization);\n  const auto cfg = h5::draft_config(proposals, slots, false, ctx, 1);')
# Portable export spelling is owned by platform.h, not an unconditional Windows declaration.
s=s.replace('__declspec(dllexport) int h5_dict8','DICT8_EXPORT int h5_dict8')
(src/'engine.cu').write_text(s)
(p/'engine-source-binding.json').write_text(json.dumps({'frozen_engine_sha256':hashlib.sha256((p/'frozen-engine.cu').read_bytes()).hexdigest(),'candidate_engine_sha256':hashlib.sha256(s.encode()).hexdigest(),'GPU_execution':False,'opt_in':'H5_GU_DICT8=1','control':'Unchanged frozen installed DLL; private source starts with exact frozen engine, not H1 derivative.'},indent=2))
print('private engine generated',len(s))
