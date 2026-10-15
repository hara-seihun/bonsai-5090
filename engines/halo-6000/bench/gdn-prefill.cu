#include "gdn-prefill.cuh"
// Reuse the independently defined FP64 scalar model and checked CUDA allocation
// utilities, not its test main. The serial device control remains unmodified.
#define main gdn_single_token_test_main
#include "gdn.cu"
#undef main
#include <chrono>

__device__ GdnView temporal_row(GdnPrefillView v,int t) {
    auto b=v.base;
    b.qkv+=int64_t(t)*v.qkv_stride;b.z+=int64_t(t)*v.z_stride;
    b.alpha+=int64_t(t)*v.alpha_stride;b.beta+=int64_t(t)*v.beta_stride;b.out+=int64_t(t)*v.out_stride;
    return b;
}
__device__ GdnScratch temporal_row(GdnPrefillScratch s,int t) {
    auto b=s.base;b.qkv+=int64_t(t)*s.qkv_stride;b.decay+=int64_t(t)*s.gate_stride;
    b.beta+=int64_t(t)*s.gate_stride;b.core+=int64_t(t)*s.core_stride;
    if(b.log_decay)b.log_decay+=int64_t(t)*s.gate_stride;
    if(b.softplus)b.softplus+=int64_t(t)*s.gate_stride;
    return b;
}
__global__ void gdn_temporal_control(GdnPrefillView v,GdnPrefillScratch s,ActivationBuffer a,int serial) {
    auto grid=cg::this_grid();
    if(serial) {
        for(int t=0;t<v.rows;++t) {
            auto b=temporal_row(v,t);auto scratch=temporal_row(s,t);auto q=a;
            q.qs+=int64_t(t)*a.q_stride;q.scale+=int64_t(t)*a.scale_stride;q.rows=1;
            gdn_conv_prepare(b,scratch);grid.sync();
            gdn_recur(b,scratch);grid.sync();gdn_finish_q8(b,scratch,q);grid.sync();
        }
    } else {
        gdn_prefill_prepare(v,s);grid.sync();
        gdn_prefill_recur(v,s);gdn_prefill_commit_history(v);grid.sync();
        gdn_prefill_finish_q8(v,s,a);grid.sync();
    }
}
struct TemporalInput {
    int rows;
    Input constant;
    std::vector<float>qkv,z,alpha,beta;
    explicit TemporalInput(int n,int scenario):rows(n),qkv(size_t(n)*ConvChannels),z(size_t(n)*Inner),alpha(size_t(n)*ValueHeads),beta(size_t(n)*ValueHeads) {
        constant.randomize(scenario);
        std::mt19937 gen(1771+scenario);std::normal_distribution<float>normal(0,.2f);
        for(auto*p:{&qkv,&z,&alpha,&beta})for(float&x:*p)x=normal(gen);
        if(scenario==2)for(int t=0;t<n;++t)for(int h=0;h<ValueHeads;++h){alpha[t*ValueHeads+h]=((h+t)&1)?80.f:-80.f;beta[t*ValueHeads+h]=((h+t)&2)?80.f:-80.f;}
        if(scenario>=3)for(int t=0;t<n;++t)for(int c=0;c<2*KeyHeads*StateDim;++c)qkv[size_t(t)*ConvChannels+c]*=scenario==3?0.f:1e-8f;
    }
    bool load(const char*path) {
        std::ifstream f(path,std::ios::binary);char magic[8];uint32_t n=0;
        f.read(magic,8);f.read((char*)&n,4);
        if(!f||std::memcmp(magic,"H6GDNP1",8)||!n||n>512)return false;
        rows=int(n);qkv.resize(size_t(n)*ConvChannels);z.resize(size_t(n)*Inner);alpha.resize(size_t(n)*ValueHeads);beta.resize(size_t(n)*ValueHeads);
        for(auto*p:{&qkv,&z,&alpha,&beta,&constant.conv,&constant.a,&constant.dt,&constant.norm,&constant.state,&constant.history})f.read((char*)p->data(),p->size()*4);
        return bool(f)&&f.peek()==EOF;
    }
    void row(int t) {
        std::copy_n(qkv.data()+size_t(t)*ConvChannels,ConvChannels,constant.qkv.data());
        std::copy_n(z.data()+size_t(t)*Inner,Inner,constant.z.data());
        std::copy_n(alpha.data()+size_t(t)*ValueHeads,ValueHeads,constant.alpha.data());
        std::copy_n(beta.data()+size_t(t)*ValueHeads,ValueHeads,constant.beta.data());
    }
};
std::vector<float> padded(const std::vector<float>&x,int rows,int width,int stride){
    std::vector<float>y(size_t(rows)*stride,-1234.25f);for(int t=0;t<rows;++t)std::copy_n(x.data()+size_t(t)*width,width,y.data()+size_t(t)*stride);return y;
}
struct TemporalGpu {
    Device mem;GdnPrefillView v;GdnPrefillScratch s;ActivationBuffer a;
    int block,grid;
    TemporalGpu(TemporalInput&i,int b,int g,bool pad):block(b),grid(g){
        int p=pad?37:0;
        v={GdnView{},i.rows,ConvChannels+p,Inner+p,ValueHeads+p,ValueHeads+p+7,Inner+p};
        v.base={mem.copy(padded(i.qkv,i.rows,ConvChannels,v.qkv_stride)),mem.copy(padded(i.z,i.rows,Inner,v.z_stride)),mem.copy(padded(i.alpha,i.rows,ValueHeads,v.alpha_stride)),mem.copy(padded(i.beta,i.rows,ValueHeads,v.beta_stride)),mem.copy(i.constant.conv),mem.copy(i.constant.a),mem.copy(i.constant.dt),mem.copy(i.constant.norm),mem.copy(i.constant.state),mem.copy(i.constant.history),nullptr};
        auto blank=[&](size_t n){return mem.copy(std::vector<float>(n,-1234.25f));};
        v.base.out=blank(size_t(i.rows)*v.out_stride);
        s={GdnScratch{},ConvChannels+p+3,ValueHeads+p+5,Inner+p+9};
        s.base={blank(size_t(i.rows)*s.qkv_stride),blank(size_t(i.rows)*s.gate_stride),blank(size_t(i.rows)*s.gate_stride),blank(size_t(i.rows)*s.core_stride),blank(size_t(i.rows)*s.gate_stride),blank(size_t(i.rows)*s.gate_stride)};
        a.k=Inner;a.rows=i.rows;a.group=32;a.q_stride=Inner+p+11;a.scale_stride=Inner/32+p+13;
        a.qs=reinterpret_cast<int8_t*>(mem.make((size_t(i.rows)*a.q_stride+3)/4));CUDA(cudaMemset(a.qs,0x5a,size_t(i.rows)*a.q_stride));
        a.scale=blank(size_t(i.rows)*a.scale_stride);
        int active=0,sm=0;CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,gdn_temporal_control,block,0));CUDA(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0));
        grid=std::min(grid,active*sm);if(grid<1)std::exit(4);
    }
    void launch(int serial,int chunk=0){
        if(!chunk)chunk=v.rows;
        for(int t=0;t<v.rows;t+=chunk){
            auto vv=v;auto ss=s;auto aa=a;vv.rows=std::min(chunk,v.rows-t);aa.rows=vv.rows;
            vv.base.qkv+=int64_t(t)*vv.qkv_stride;vv.base.z+=int64_t(t)*vv.z_stride;vv.base.alpha+=int64_t(t)*vv.alpha_stride;vv.base.beta+=int64_t(t)*vv.beta_stride;vv.base.out+=int64_t(t)*vv.out_stride;
            ss.base.qkv+=int64_t(t)*ss.qkv_stride;ss.base.core+=int64_t(t)*ss.core_stride;ss.base.decay+=int64_t(t)*ss.gate_stride;ss.base.beta+=int64_t(t)*ss.gate_stride;ss.base.log_decay+=int64_t(t)*ss.gate_stride;ss.base.softplus+=int64_t(t)*ss.gate_stride;
            aa.qs+=int64_t(t)*aa.q_stride;aa.scale+=int64_t(t)*aa.scale_stride;
            void*args[]={&vv,&ss,&aa,&serial};CUDA(cudaLaunchCooperativeKernel((void*)gdn_temporal_control,dim3(grid),dim3(block),args,0,nullptr));
        }
    }
    std::vector<float>matrix(const float*p,int rows,int width,int stride){
        auto all=read_device(p,size_t(rows)*stride);std::vector<float>out(size_t(rows)*width);
        for(int t=0;t<rows;++t){std::copy_n(all.data()+size_t(t)*stride,width,out.data()+size_t(t)*width);for(int c=width;c<stride;++c)if(all[size_t(t)*stride+c]!=-1234.25f){std::fprintf(stderr,"padding overwritten\n");std::exit(9);}}return out;
    }
    Output output(){
        CUDA(cudaDeviceSynchronize());int n=v.rows;
        Output o{matrix(s.base.qkv,n,ConvChannels,s.qkv_stride),matrix(s.base.decay,n,ValueHeads,s.gate_stride),matrix(s.base.beta,n,ValueHeads,s.gate_stride),matrix(s.base.core,n,Inner,s.core_stride),matrix(v.base.out,n,Inner,v.out_stride),read_device(v.base.state,ValueHeads*ValueDim*StateDim),read_device(v.base.history,ConvChannels*3),matrix(a.scale,n,Inner/32,a.scale_stride),std::vector<int8_t>(size_t(n)*Inner),matrix(s.base.log_decay,n,ValueHeads,s.gate_stride),matrix(s.base.softplus,n,ValueHeads,s.gate_stride)};
        std::vector<int8_t>q(size_t(n)*a.q_stride);CUDA(cudaMemcpy(q.data(),a.qs,q.size(),cudaMemcpyDeviceToHost));
        for(int t=0;t<n;++t){std::copy_n(q.data()+size_t(t)*a.q_stride,Inner,o.qs.data()+size_t(t)*Inner);for(int c=Inner;c<a.q_stride;++c)if(q[size_t(t)*a.q_stride+c]!=0x5a)std::exit(9);}return o;
    }
};
Output temporal_run(TemporalInput&i,int mode,int block,int grid,bool pad,int chunk=0){TemporalGpu gpu(i,block,grid,pad);gpu.launch(mode,chunk);return gpu.output();}
Error row_error(const std::vector<float>&x,int t,const std::vector<double>&ref){return difference(std::vector<float>(x.begin()+size_t(t)*ref.size(),x.begin()+size_t(t+1)*ref.size()),ref);}
bool cpu_check(TemporalInput&i,const Output&o,double&rel,double&abs){
    Reference ref(i.constant);bool pass=true;
    auto check=[&](Error e){rel=std::max(rel,e.relative);abs=std::max(abs,e.absolute);pass&=e.relative<3e-5||e.absolute<1e-7;};
    for(int t=0;t<i.rows;++t){i.row(t);ref.step(i.constant);check(row_error(o.qkv,t,ref.qkv));check(row_error(o.decay,t,ref.decay));check(row_error(o.beta,t,ref.beta));check(row_error(o.core,t,ref.core));check(row_error(o.out,t,ref.out));check(row_error(o.log_decay,t,ref.log_decay));check(row_error(o.softplus,t,ref.softplus));}
    check(difference(o.state,ref.state));check(difference(o.history,ref.history));return pass;
}
bool guard(TemporalInput&i,int scenario){
    auto serial=temporal_run(i,1,128,64,false),prefill=temporal_run(i,0,256,128,true);
    auto other=temporal_run(i,0,128,64,false),chunks=temporal_run(i,0,256,128,true,2);
    bool exact=exact_equal(serial,prefill),schedule=exact_equal(serial,other),split=exact_equal(serial,chunks);
    double rel=0,abs=0;bool cpu=cpu_check(i,prefill,rel,abs);
    if(!exact){const char*names[]={"qkv","decay","beta","core","out","state","history","scales","log_decay","softplus"};const std::vector<float>*a[]={&serial.qkv,&serial.decay,&serial.beta,&serial.core,&serial.out,&serial.state,&serial.history,&serial.scales,&serial.log_decay,&serial.softplus};const std::vector<float>*b[]={&prefill.qkv,&prefill.decay,&prefill.beta,&prefill.core,&prefill.out,&prefill.state,&prefill.history,&prefill.scales,&prefill.log_decay,&prefill.softplus};for(int j=0;j<10;++j)if(!bit_equal(*a[j],*b[j]))for(size_t k=0;k<a[j]->size();++k)if(std::memcmp(&(*a[j])[k],&(*b[j])[k],4)){std::fprintf(stderr,"first %s mismatch[%zu]: %.9g vs %.9g\n",names[j],k,(*a[j])[k],(*b[j])[k]);break;}}
    std::printf("{\"scenario\":%d,\"rows\":%d,\"serial_bits\":%s,\"schedule_bits\":%s,\"chunk2_bits\":%s,\"cpu_equal\":%s,\"cpu_max_relative\":%.9g,\"cpu_max_absolute\":%.9g}\n",scenario,i.rows,exact?"true":"false",schedule?"true":"false",split?"true":"false",cpu?"true":"false",rel,abs);std::fflush(stdout);
    return exact&&schedule&&split&&cpu;
}
void timing(int rows){
    TemporalInput i(rows,0);TemporalGpu gpu(i,256,128,false);
    cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
    for(int mode:{1,0}){for(int j=0;j<3;++j)gpu.launch(mode);CUDA(cudaDeviceSynchronize());CUDA(cudaEventRecord(start));for(int j=0;j<10;++j)gpu.launch(mode);CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));std::printf("{\"rows\":%d,\"mode\":\"%s\",\"ms\":%.9g,\"block\":256,\"grid\":%d,\"iterations\":10}\n",rows,mode?"serial":"prefill",ms/10,gpu.grid);}
    CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));
}
int main(int argc,char**argv){
    if(argc==3&&std::string(argv[1])=="--fixture"){TemporalInput i(1,0);if(!i.load(argv[2])){std::fprintf(stderr,"invalid H6GDNP1 fixture\n");return 6;}return guard(i,-1)?0:1;}
    if(argc==2&&std::string(argv[1])=="--timing"){for(int n:{1,8,32,128,512})timing(n);return 0;}
    bool pass=true;for(int n:{1,2,8,32,128,512})for(int scenario=0;scenario<5;++scenario){TemporalInput i(n,scenario);pass&=guard(i,scenario);}
    std::printf("{\"passed\":%s,\"cases\":30,\"comparison\":\"memcmp including signed zero, quant bytes, scales, all prepared intermediates and final state/history\"}\n",pass?"true":"false");return pass?0:1;
}
