#include "qwen3.hpp"
#include "common.hpp"
#include <cstdlib>
#include <iostream>

namespace {
bool graph_default() {
    const char* v=std::getenv("EDGE_INFER_ENABLE_QWEN3_GRAPH");
    return v && v[0]=='1' && v[1]=='\0';
}
TensorView<const uint32_t,1> tokens_from(const Tensor<uint32_t>* in) {
    if(!in || in->device()!=Device::CUDA) throw std::invalid_argument("Session requires CUDA tokens");
    return borrow_tensor_view<1>(*in);
}
}
template <typename T>
Qwen3Session<T>::Qwen3Session(std::shared_ptr<const Qwen3Model<T>> m)
    : Qwen3Session(std::move(m),graph_default()) {}
template <typename T>
Qwen3Session<T>::Qwen3Session(std::shared_ptr<const Qwen3Model<T>> m,bool graph)
    : model_(std::move(m)),use_cuda_graph_(graph) {
    if(!model_) throw std::invalid_argument("Session requires a prepared model");
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    if(dev!=model_->cuda_device_id()) throw std::invalid_argument("Session must be created on its model CUDA device");
    try { initialize(); } catch(...) { release(); throw; }
}
template <typename T>
Qwen3Session<T>::Qwen3Session(const typename Qwen3Model<T>::Parameters& p,const ModelConfig& c)
    : Qwen3Session(std::make_shared<Qwen3Model<T>>(p,c)) {}
template <typename T>
Qwen3Session<T>::Qwen3Session(const typename Qwen3Model<T>::Parameters& p,
    const typename Qwen3Model<T>::IntegerParameters& q,const typename Qwen3Model<T>::Parameters& s,
    const typename Qwen3Model<T>::IntegerParameters& z,const ModelConfig& c)
    : Qwen3Session(std::make_shared<Qwen3Model<T>>(p,q,s,z,c)) {}
template <typename T> void Qwen3Session<T>::initialize() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&context_.stream,cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&context_.handle));
    context_=op::cuda::prepare_execution_context(context_.handle,context_.stream);
    op::cuda::bind_execution_context(context_);
    decode_plan_=plan_decoder_workspace<T>(model_->config(),1);
    decode_buffers_=resolve_decoder_buffers<T>(model_->config(),1,decode_plan_,decode_workspace_);
    graph_storage_.reserve(1280+5*sizeof(T*));
    graph_input_={graph_storage_.template ptr_at<uint32_t>(0),{1},{1}};
    device_offset_=graph_storage_.template ptr_at<size_t>(256);
    device_lengths_=graph_storage_.template ptr_at<int>(512);
    device_pingpong_=graph_storage_.template ptr_at<int>(768);
    graph_branches_=graph_storage_.template ptr_at<T*>(1024);
    CUDA_CHECK(cudaMemsetAsync(device_pingpong_,0,sizeof(int),context_.stream));
    const auto& c=model_->config();
    T* branches[5];
    for(size_t i=0;i<5;++i) branches[i]=decode_buffers_.attention_scratch.data+i*c.n_heads*(c.head_dim+2);
    CUDA_CHECK(cudaMemcpyAsync(graph_branches_,branches,sizeof(branches),cudaMemcpyHostToDevice,context_.stream));
    sampling_plan_=op::cuda::prepare_sampling(context_,c.vocab_size);
    sampling_storage_.reserve(sampling_plan_.total_bytes+512);
    sampling_scratch_={sampling_storage_.template ptr_at<unsigned char>(0),{sampling_plan_.total_bytes},{1}};
    sampled_token_={sampling_storage_.template ptr_at<uint32_t>(sampling_plan_.total_bytes),{1},{1}};
    sampled_probability_={sampling_storage_.template ptr_at<float>(sampling_plan_.total_bytes+256),{1},{1}};
    host_ids_.resize(1); bound_bases_.reserve(c.n_layers); synchronize();
}
template <typename T> void Qwen3Session<T>::release() noexcept {
    if(!model_) return;
    int prev=model_->cuda_device_id(); cudaGetDevice(&prev);
    if(prev!=model_->cuda_device_id()) cudaSetDevice(model_->cuda_device_id());
    if(context_.stream) cudaStreamSynchronize(context_.stream);
    if(graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_=nullptr; }
    if(graph_) { cudaGraphDestroy(graph_); graph_=nullptr; }
    managed_cache_.reset();
    decode_workspace_.release(); prefill_workspace_.release(); graph_storage_.release(); sampling_storage_.release();
    if(context_.handle) { cublasDestroy(context_.handle); context_.handle=nullptr; }
    if(context_.stream) { cudaStreamDestroy(context_.stream); context_.stream=nullptr; }
    if(prev!=model_->cuda_device_id()) cudaSetDevice(prev);
}
template <typename T> Qwen3Session<T>::~Qwen3Session() { release(); }
template <typename T> void Qwen3Session<T>::synchronize() const {
    if(context_.stream) CUDA_CHECK(cudaStreamSynchronize(context_.stream));
}
template <typename T> void Qwen3Session<T>::set_graph_enabled(bool enabled) { synchronize(); use_cuda_graph_=enabled; }
template <typename T> Qwen3Session<T>& Qwen3Session<T>::cuda() { return *this; }
template <typename T> Qwen3Session<T>& Qwen3Session<T>::cpu() { throw std::runtime_error("Prepared CUDA sessions cannot migrate to CPU"); }
template <typename T> void Qwen3Session<T>::print_model_info() const {
    std::cout << "Prepared decoder: " << get_n_layers() << " layers, " << get_hidden_size()
              << " hidden, " << get_vocab_size() << " vocabulary; workspace " << decode_workspace_bytes() << " bytes\n";
}
template <typename T> size_t Qwen3Session<T>::estimate_prefill_workspace_bytes(size_t rows) const {
    return plan_decoder_workspace<T>(model_->config(),rows).total_bytes();
}
template <typename T> std::unique_ptr<Qwen3Session<T>> Qwen3Session<T>::create(
    std::shared_ptr<const Qwen3Model<T>> m,size_t n,bool graph) {
    if(!m || !n || n>m->config().max_position_embeddings) throw std::invalid_argument("Session capacity must be positive and within the model limit");
    auto s=std::make_unique<Qwen3Session<T>>(std::move(m),graph);
    s->managed_cache_=std::make_unique<KVCache<T>>(s->get_n_layers(),n,s->get_n_kv_heads()*s->get_head_dim(),Device::CUDA);
    return s;
}
template <typename T> std::unique_ptr<Qwen3Session<T>> Qwen3Session<T>::new_session(size_t n,bool graph) const { return create(model_,n,graph); }
template <typename T> void Qwen3Session<T>::require_managed() const {
    if(!managed_cache_) throw std::logic_error("Context operations require create or new_session");
}
template <typename T> size_t Qwen3Session<T>::context_size() const { require_managed(); return managed_cache_->size(); }
template <typename T> size_t Qwen3Session<T>::context_capacity() const { require_managed(); return managed_cache_->get_max_seq_len(); }
template <typename T> void Qwen3Session<T>::reset() { require_managed(); synchronize(); managed_cache_->clear(); history_ready_=false; }
template <typename T> void Qwen3Session<T>::validate_and_bind(TensorView<const uint32_t,1> in,KVCache<T>* cache,bool decode) {
    if(!cache || !in.data || !in.shape[0] || in.stride[0]!=1 || (decode && in.shape[0]!=1))
        throw std::invalid_argument("Session requires nonempty contiguous CUDA tokens; decode takes one token");
    if(managed_cache_ && managed_cache_.get()!=cache) throw std::invalid_argument("Managed session requires its own cache");
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    if(dev!=model_->cuda_device_id()) throw std::invalid_argument("Session CUDA device changed");
    const auto& c=model_->config();
    if(cache->device()!=Device::CUDA || cache->get_n_layers()!=c.n_layers || cache->get_head_dim()!=c.n_kv_heads*c.head_dim ||
       !cache->get_max_seq_len() || cache->get_max_seq_len()>c.max_position_embeddings ||
       cache->size()<in.shape[0] || cache->size()>cache->get_max_seq_len())
        throw std::invalid_argument("Session cache shape, device or extent is incompatible");
    auto check_pointer=[&](const void* ptr) {
        cudaPointerAttributes a{}; auto status=cudaPointerGetAttributes(&a,ptr);
        if(status!=cudaSuccess) { cudaGetLastError(); throw std::invalid_argument("Invalid CUDA session pointer"); }
        if(a.type!=cudaMemoryTypeDevice || a.device!=dev) throw std::invalid_argument("Session pointers must reside on model CUDA device");
    };
    check_pointer(in.data);
    if(bound_cache_ && (bound_cache_!=cache || bound_capacity_!=cache->get_max_seq_len()))
        throw std::invalid_argument("Session cache binding cannot change");
    for(size_t i=0;i<c.n_layers;++i) {
        auto k=cache->k_capacity_view(i),v=cache->v_capacity_view(i);
        if(bound_cache_) {
            if(bound_bases_[i].first!=k.data || bound_bases_[i].second!=v.data) throw std::invalid_argument("Session cache storage changed");
        } else { check_pointer(k.data); check_pointer(v.data); }
    }
    if(host_ids_.size()<in.shape[0]) host_ids_.resize(in.shape[0]);
    CUDA_CHECK(cudaMemcpyAsync(host_ids_.data(),in.data,in.shape[0]*sizeof(uint32_t),cudaMemcpyDeviceToHost,context_.stream));
    synchronize();
    for(size_t i=0;i<in.shape[0];++i) if(host_ids_[i]>=c.vocab_size) throw std::invalid_argument("Session token id exceeds vocabulary");
    if(!bound_cache_) {
        for(size_t i=0;i<c.n_layers;++i) bound_bases_.emplace_back(cache->k_capacity_view(i).data,cache->v_capacity_view(i).data);
        bound_cache_=cache; bound_capacity_=cache->get_max_seq_len();
    }
}
template <typename T> void Qwen3Session<T>::run(TensorView<const uint32_t,1> in,KVCache<T>& cache,DecoderBuffers<T>& b,const DecoderStep& step) {
    op::cuda::gather<T>(context_,in,model_->embedding_view(),b.residual);
    auto h=execute_decoder<T>(context_,*model_,cache,b,step,graph_branches_);
    execute_linear<T>(context_,h.as_const(),model_->output_weight(),b.logits);
}
template <typename T> void Qwen3Session<T>::capture_graph(KVCache<T>& cache) {
    DecoderStep step{host_offset_,host_offset_,DecoderMode::Graph,device_offset_,device_offset_,device_lengths_,device_pingpong_};
    op::cuda::bind_execution_context(context_);
    run(graph_input_.as_const(),cache,decode_buffers_,step); synchronize();
    CUDA_CHECK(cudaStreamBeginCapture(context_.stream,cudaStreamCaptureModeThreadLocal));
    bool capturing=true;
    try {
        run(graph_input_.as_const(),cache,decode_buffers_,step);
        auto status=cudaStreamEndCapture(context_.stream,&graph_); capturing=false; CUDA_CHECK(status);
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_,graph_,nullptr,nullptr,0));
    } catch(...) {
        if(capturing) cudaStreamEndCapture(context_.stream,&graph_);
        if(graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_=nullptr; }
        if(graph_) { cudaGraphDestroy(graph_); graph_=nullptr; }
        throw;
    }
}
template <typename T> TensorView<T,2> Qwen3Session<T>::execute(TensorView<const uint32_t,1> in,KVCache<T>& cache,bool prefill,bool graph) {
    validate_and_bind(in,&cache,!prefill);
    const size_t rows=in.shape[0],offset=cache.size()-rows;
    if(prefill) {
        if(prefill_rows_!=rows) {
            const auto plan=plan_decoder_workspace<T>(model_->config(),rows);
            prefill_buffers_=resolve_decoder_buffers<T>(model_->config(),rows,plan,prefill_workspace_); prefill_rows_=rows;
        }
        op::cuda::bind_execution_context(context_); run(in,cache,prefill_buffers_,{offset,offset,DecoderMode::Prefill});
        synchronize(); return prefill_buffers_.logits;
    }
    if(graph) {
        host_offset_=offset; host_length_=static_cast<int>(cache.size());
        if(in.data!=graph_input_.data) CUDA_CHECK(cudaMemcpyAsync(graph_input_.data,in.data,sizeof(uint32_t),cudaMemcpyDeviceToDevice,context_.stream));
        CUDA_CHECK(cudaMemcpyAsync(device_offset_,&host_offset_,sizeof(size_t),cudaMemcpyHostToDevice,context_.stream));
        CUDA_CHECK(cudaMemcpyAsync(device_lengths_,&host_length_,sizeof(int),cudaMemcpyHostToDevice,context_.stream));
        if(!graph_exec_) capture_graph(cache);
        CUDA_CHECK(cudaGraphLaunch(graph_exec_,context_.stream));
    } else { op::cuda::bind_execution_context(context_); run(in,cache,decode_buffers_,{offset,offset,DecoderMode::Decode}); }
    synchronize(); return decode_buffers_.logits;
}
template <typename T> TensorView<T,2> Qwen3Session<T>::prefill(TensorView<const uint32_t,1> in) {
    require_managed();
    if(!in.shape[0]) throw std::invalid_argument("Prompt must be nonempty");
    if(in.shape[0]>context_capacity()) throw std::length_error("Prompt exceeds session capacity");
    const size_t previous=context_size();
    try { managed_cache_->resize(in.shape[0]); auto out=execute(in,*managed_cache_,true,false); history_ready_=true; return out; }
    catch(...) { managed_cache_->resize(previous); throw; }
}
template <typename T> TensorView<T,2> Qwen3Session<T>::decode(TensorView<const uint32_t,1> in) {
    require_managed();
    if(!history_ready_) throw std::logic_error("Decode requires successful prefill");
    if(in.shape[0]!=1) throw std::invalid_argument("Decode takes one token");
    const size_t previous=context_size();
    if(previous>=context_capacity()) throw std::length_error("Session context capacity exhausted");
    try { managed_cache_->resize(previous+1); return execute(in,*managed_cache_,false,use_cuda_graph_); }
    catch(...) { managed_cache_->resize(previous); throw; }
}
template <typename T> TensorView<T,2> Qwen3Session<T>::decode(uint32_t token) {
    if(token>=get_vocab_size()) throw std::invalid_argument("Token exceeds vocabulary");
    CUDA_CHECK(cudaMemcpyAsync(graph_input_.data,&token,sizeof(token),cudaMemcpyHostToDevice,context_.stream));
    return decode(graph_input_.as_const());
}
template <typename T> TensorView<T,2> Qwen3Session<T>::forward_eager(const Tensor<uint32_t>* in,KVCache<T>* kv) {
    if(!kv) throw std::invalid_argument("Session requires typed cache"); return execute(tokens_from(in),*kv,false,false);
}
template <typename T> TensorView<T,2> Qwen3Session<T>::prefill_eager(const Tensor<uint32_t>* in,KVCache<T>* kv) {
    if(!kv) throw std::invalid_argument("Session requires typed cache"); return execute(tokens_from(in),*kv,true,false);
}
template <typename T> TensorView<T,2> Qwen3Session<T>::forward_for_graph_logits_only(const Tensor<uint32_t>* in,KVCache<T>* kv) {
    if(!kv) throw std::invalid_argument("Session requires typed cache"); return execute(tokens_from(in),*kv,false,true);
}
template <typename T> uint32_t* Qwen3Session<T>::sample_logits(TensorView<const T,2> logits,float temp,float p,size_t k,curandState* states) {
    op::cuda::sample<T>(context_,logits,sampled_token_,sampled_probability_,sampling_scratch_,sampling_plan_,temp,p,k,states);
    synchronize(); return sampled_token_.data;
}
template <typename T> uint32_t* Qwen3Session<T>::forward(const Tensor<uint32_t>* in,ThreadPool&,KVCacheBase* base,size_t k,float temp,float p,curandState* states) {
    op::cuda::validate_sampling_policy(get_vocab_size(),temp,p,k,states!=nullptr);
    auto* kv=dynamic_cast<KVCache<T>*>(base);
    auto logits=use_cuda_graph_ ? forward_for_graph_logits_only(in,kv) : forward_eager(in,kv);
    return sample_logits(logits.as_const(),temp,p,k,states);
}
template <typename T> uint32_t* Qwen3Session<T>::prefill(const Tensor<uint32_t>* in,ThreadPool&,KVCacheBase* base,size_t k,float temp,float p,curandState* states) {
    op::cuda::validate_sampling_policy(get_vocab_size(),temp,p,k,states!=nullptr);
    auto logits=prefill_eager(in,dynamic_cast<KVCache<T>*>(base));
    return sample_logits(logits.subview({logits.shape[0]-1,0},{1,logits.shape[1]}).as_const(),temp,p,k,states);
}
template class Qwen3Session<__nv_bfloat16>;
template class Qwen3Session<float>;
