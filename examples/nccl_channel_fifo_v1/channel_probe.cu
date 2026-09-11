// Paired timers with fixed CPU placement and explicit green-context controls.
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>
#include <sched.h>
#include <pthread.h>
#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(2); } } while(0)
#define NCCL(x) do { auto e=(x); if(e!=ncclSuccess) { std::fprintf(stderr,"NCCL: %s\n",ncclGetErrorString(e)); std::exit(3); } } while(0)
#define DRIVER(x) do { auto e=(x); if(e!=CUDA_SUCCESS) { const char* s=nullptr; cuGetErrorString(e,&s); std::fprintf(stderr,"DRIVER: %s: %s\n",#x,s?s:"unknown"); std::exit(6); } } while(0)
__global__ void sample_sms(unsigned* seen) {
  unsigned sm; asm volatile("mov.u32 %0, %smid;" : "=r"(sm));
  if(threadIdx.x==0 && sm<4096) atomicExch(seen+sm,1);
  unsigned long long begin=clock64();
  while(clock64()-begin<2000) {}
}
class Barrier {
  std::mutex mutex; std::condition_variable cv; int count, waiting=0, generation=0;
 public:
  explicit Barrier(int n):count(n){}
  void wait() { std::unique_lock<std::mutex> lock(mutex); int g=generation;
    if(++waiting==count) {waiting=0; ++generation; cv.notify_all();}
    else cv.wait(lock,[&]{return generation!=g;}); }
};
__global__ void fill(float* p,size_t n,float value) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) p[i]=value;
}
__global__ void verify(const float* p,size_t n,float expected,int* bad) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
    if(p[i]!=expected) atomicExch(bad,1);
}
int main(int argc,char** argv) {
  if(argc!=10) { std::fprintf(stderr,"usage: probe sizes output width warmup timed persistent rotate allocation_mib requested_sms\n"); return 1; }
  std::ifstream input(argv[1]); std::vector<size_t> sizes; size_t size;
  while(input>>size) sizes.push_back(size);
  int width=std::atoi(argv[3]),warm=std::atoi(argv[4]),timed=std::atoi(argv[5]);
  bool persistent=std::atoi(argv[6]),rotate=std::atoi(argv[7]);
  size_t allocation=std::strtoull(argv[8],nullptr,10)<<20;
  int requested_sms=std::atoi(argv[9]);
  DRIVER(cuInit(0));
  cpu_set_t affinity; CPU_ZERO(&affinity);
  if(sched_getaffinity(0,sizeof(affinity),&affinity)!=0) return 7;
  std::vector<int> cpus;
  for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&affinity)) cpus.push_back(cpu);
  if(cpus.size()<size_t(width) || requested_sms<0) return 7;
  auto pin=[&](int r) { cpu_set_t one;CPU_ZERO(&one);CPU_SET(cpus[r],&one);
    if(pthread_setaffinity_np(pthread_self(),sizeof(one),&one)!=0) std::exit(7); };
  if(sizes.empty() || (width!=2 && width!=4) || warm<1 || timed<1 || allocation<*std::max_element(sizes.begin(),sizes.end())) return 4;
  std::vector<int> devs(width); for(int r=0;r<width;++r) devs[r]=r;
  std::vector<ncclComm_t> comms(width); NCCL(ncclCommInitAll(comms.data(),width,devs.data()));
  std::vector<float*> send(width),recv(width); std::vector<int*> bad(width);
  std::vector<cudaStream_t> streams(width);
#if CUDA_VERSION >= 12040
  std::vector<CUgreenCtx> green(width,nullptr);
#endif
  std::vector<int> granted(width),observed(width);
  for(int r=0;r<width;++r) {
    CUDA(cudaSetDevice(r)); CUDA(cudaMalloc(&send[r],allocation)); CUDA(cudaMalloc(&recv[r],allocation));
    CUDA(cudaMalloc(&bad[r],sizeof(int)));
    cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,r));granted[r]=prop.multiProcessorCount;
    if(requested_sms) {
#if CUDA_VERSION >= 12040
      CUcontext saved,converted;DRIVER(cuCtxGetCurrent(&saved));
      CUdevResource input,part,unused,check;unsigned count=1;CUdevResourceDesc descriptor;
      DRIVER(cuDeviceGetDevResource(r,&input,CU_DEV_RESOURCE_TYPE_SM));
      DRIVER(cuDevSmResourceSplitByCount(&part,&count,&input,&unused,0,requested_sms));
      if(count!=1) return 6;
      DRIVER(cuDevResourceGenerateDesc(&descriptor,&part,1));
      DRIVER(cuGreenCtxCreate(&green[r],descriptor,r,CU_GREEN_CTX_DEFAULT_STREAM));
      DRIVER(cuGreenCtxGetDevResource(green[r],&check,CU_DEV_RESOURCE_TYPE_SM));
      granted[r]=check.sm.smCount;
      DRIVER(cuCtxFromGreenCtx(&converted,green[r]));DRIVER(cuCtxSetCurrent(converted));
      CUDA(cudaStreamCreateWithFlags(&streams[r],cudaStreamNonBlocking));
      DRIVER(cuCtxSetCurrent(saved));
#else
      std::fprintf(stderr,"UNSUPPORTED: this toolkit has no green contexts\n");return 6;
#endif
    } else CUDA(cudaStreamCreateWithFlags(&streams[r],cudaStreamNonBlocking));
    unsigned* seen;CUDA(cudaMalloc(&seen,4096*sizeof(unsigned)));CUDA(cudaMemset(seen,0,4096*sizeof(unsigned)));
    sample_sms<<<4096,128,0,streams[r]>>>(seen);CUDA(cudaStreamSynchronize(streams[r]));
    std::vector<unsigned> sampled(4096);CUDA(cudaMemcpy(sampled.data(),seen,4096*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for(unsigned value:sampled) observed[r]+=value;CUDA(cudaFree(seen));
    std::fprintf(stdout,"META rank=%d requested_sms=%d granted_sms=%d observed_sms=%d stream=%p cpu=%d\n",
                 r,requested_sms,granted[r],observed[r],reinterpret_cast<void*>(streams[r]),cpus[r]);
    if(observed[r]>granted[r] || !observed[r] || granted[r]<requested_sms) return 6;
    fill<<<256,256>>>(send[r],allocation/4,float(r+1)); CUDA(cudaMemset(recv[r],0,allocation)); CUDA(cudaDeviceSynchronize());
  }
  std::vector<std::vector<double>> events(sizes.size(),std::vector<double>(width)),wall=events;
  std::vector<std::vector<int>> errors(sizes.size(),std::vector<int>(width));
  Barrier barrier(width);
  auto cell=[&](int r,size_t k) {
    size_t bytes=sizes[k],stride=((bytes+255)/256)*256,slots=allocation/stride;
    cudaEvent_t begin,end; CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    auto offset=[&](int it){return rotate ? (size_t(it)%slots)*stride/4 : 0;};
    auto issue=[&](int it){size_t o=offset(it); NCCL(ncclAllReduce(send[r]+o,recv[r]+o,bytes/4,ncclFloat,ncclSum,comms[r],streams[r]));};
    for(int it=0;it<warm;++it) issue(it);
    CUDA(cudaStreamSynchronize(streams[r])); barrier.wait();
    auto t0=std::chrono::steady_clock::now();
    CUDA(cudaEventRecord(begin,streams[r]));
    for(int it=0;it<timed;++it) issue(it);
    CUDA(cudaEventRecord(end,streams[r])); CUDA(cudaStreamSynchronize(streams[r]));
    auto t1=std::chrono::steady_clock::now();
    float ms; CUDA(cudaEventElapsedTime(&ms,begin,end));
    events[k][r]=ms*1000.0/timed;
    wall[k][r]=std::chrono::duration<double,std::micro>(t1-t0).count()/timed;
    CUDA(cudaMemsetAsync(bad[r],0,sizeof(int),streams[r]));
    verify<<<256,256,0,streams[r]>>>(recv[r]+offset(timed-1),bytes/4,float(width*(width+1)/2),bad[r]);
    CUDA(cudaMemcpyAsync(&errors[k][r],bad[r],sizeof(int),cudaMemcpyDeviceToHost,streams[r]));
    CUDA(cudaStreamSynchronize(streams[r])); CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end)); barrier.wait();
  };
  if(persistent) {
    std::vector<std::thread> threads;
    for(int r=0;r<width;++r) threads.emplace_back([&,r]{pin(r);CUDA(cudaSetDevice(r));for(size_t k=0;k<sizes.size();++k) cell(r,k);});
    for(auto& t:threads) t.join();
  } else for(size_t k=0;k<sizes.size();++k) {
    std::vector<std::thread> threads;
    for(int r=0;r<width;++r) threads.emplace_back([&,r]{pin(r);CUDA(cudaSetDevice(r));cell(r,k);});
    for(auto& t:threads) t.join();
  }
  std::ofstream output(argv[2]); output<<"bytes,width,warmup,timed,persistent,rotate,allocation_bytes,event_us,wall_us,mismatching_ranks,rank,requested_sms,granted_sms,observed_sms\n"; output.precision(12);
  int failures=0;
  for(size_t k=0;k<sizes.size();++k) {
    int count=0;for(int e:errors[k])count+=e;failures+=count;
    output<<sizes[k]<<','<<width<<','<<warm<<','<<timed<<','<<persistent<<','<<rotate<<','<<allocation<<','<<*std::max_element(events[k].begin(),events[k].end())<<','<<*std::max_element(wall[k].begin(),wall[k].end())<<','<<count<<",-1,"<<requested_sms<<','<<*std::min_element(granted.begin(),granted.end())<<','<<*std::max_element(observed.begin(),observed.end())<<'\n';
    for(int r=0;r<width;++r) output<<sizes[k]<<','<<width<<','<<warm<<','<<timed<<','<<persistent<<','<<rotate<<','<<allocation<<','<<events[k][r]<<','<<wall[k][r]<<','<<errors[k][r]<<','<<r<<','<<requested_sms<<','<<granted[r]<<','<<observed[r]<<'\n';
  }
  for(int r=0;r<width;++r) { CUDA(cudaSetDevice(r)); CUDA(cudaStreamDestroy(streams[r])); CUDA(cudaFree(send[r])); CUDA(cudaFree(recv[r])); CUDA(cudaFree(bad[r])); NCCL(ncclCommDestroy(comms[r])); }
#if CUDA_VERSION >= 12040
  for(auto context:green) if(context) DRIVER(cuGreenCtxDestroy(context));
#endif
  return failures ? 5 : 0;
}
