// qwen.cpp
#include "qwen.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "common.hpp"
#include "decoder_executor.hpp"
#include "execution/program.hpp"
#include "graph_runner.hpp"

// -------------------------------
// QwenModel<T> 构造函数
// -------------------------------
template <typename T>
QwenModel<T>::QwenModel(const std::unordered_map<std::string, Tensor<T>> &params,
                        const ModelConfig &config)
    : params_(params) {
    // 从 config 中提取基本参数
    vocab_size_ = config.at("vocab_size");
    n_layers_ = config.at("n_layers");
    n_heads_ = config.at("n_heads");
    n_kv_heads_ = config.at("n_kv_heads");
    hidden_size_ = config.at("hidden_size");
    intermediate_size_ = config.at("intermediate_size");
    max_position_embeddings_ = config.at("max_position_embeddings");
    bos_token_id_ = static_cast<uint32_t>(config.at("bos_token_id"));
    eos_token_id_ = static_cast<uint32_t>(config.at("eos_token_id"));
    rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps"));
    rope_theta_ = static_cast<float>(config.at("rope_theta"));
    head_dim_ = hidden_size_ / n_heads_;

    // 检查是否有量化类型参数
    if (config.find("quant_type") != config.end()) {
        quant_type_ = config.at("quant_type");
    }

    // 检查是否有分组大小参数
    if (config.find("group_size") != config.end()) {
        group_size_ = config.at("group_size");
    }

    compute_streams_.fill(nullptr);
    fa_done_events_.fill(nullptr);
    graph_runtime_ = std::make_unique<CudaGraphRuntime<T>>();
    decode_workspace_ = std::make_unique<CudaWorkspaceArena>();

    device_ = params_.empty() ? Device::CPU : params_.begin()->second.device();
    use_cuda_graph_ = use_cuda_graph_ && device_ == Device::CUDA && quant_type_ == 0;
    sample_mode_ = device_ == Device::CUDA ? SampleMode::GPU : SampleMode::CPU;

    operators_ = std::make_unique<op::UnifiedOperators<T>>(device_);
    cpu_operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CPU);

    if (device_ == Device::CUDA) {
        initialize_cuda_runtime();
        initialize_decode_workspace();
    }
}

// 带量化参数的构造函数
template <typename T>
QwenModel<T>::QwenModel(const std::unordered_map<std::string, Tensor<T>> &params,
                        const std::unordered_map<std::string, Tensor<int32_t>> &qweight_params,
                        const std::unordered_map<std::string, Tensor<T>> &scales_params,
                        const std::unordered_map<std::string, Tensor<int32_t>> &qzeros_params,
                        const ModelConfig &config)
    : params_(params), qweight_params_(qweight_params), scales_params_(scales_params), qzeros_params_(qzeros_params) {
    // 从 config 中提取基本参数
    vocab_size_ = config.at("vocab_size");
    n_layers_ = config.at("n_layers");
    n_heads_ = config.at("n_heads");
    n_kv_heads_ = config.at("n_kv_heads");
    hidden_size_ = config.at("hidden_size");
    intermediate_size_ = config.at("intermediate_size");
    max_position_embeddings_ = config.at("max_position_embeddings");
    bos_token_id_ = static_cast<uint32_t>(config.at("bos_token_id"));
    eos_token_id_ = static_cast<uint32_t>(config.at("eos_token_id"));
    rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps"));
    rope_theta_ = static_cast<float>(config.at("rope_theta"));
    head_dim_ = hidden_size_ / n_heads_;

    // 设置量化类型为AWQ
    quant_type_ = 1;

    // 检查是否有分组大小参数
    if (config.find("group_size") != config.end()) {
        group_size_ = config.at("group_size");
    }
    // 创建非const副本并移动到CUDA
    for (const auto &kv : qweight_params) {
        if (kv.second.device() != Device::CUDA) {
            // 创建副本并移动到CUDA
            Tensor<int32_t> tensor_copy = kv.second;
            tensor_copy.cuda();
            qweight_params_[kv.first] = tensor_copy;
        }
    }
    for (const auto &kv : scales_params) {
        if (kv.second.device() != Device::CUDA) {
            // 创建副本并移动到CUDA
            Tensor<T> tensor_copy = kv.second;
            tensor_copy.cuda();
            scales_params_[kv.first] = tensor_copy;
        }
    }
    for (const auto &kv : qzeros_params) {
        if (kv.second.device() != Device::CUDA) {
            // 创建副本并移动到CUDA
            Tensor<int32_t> tensor_copy = kv.second;
            tensor_copy.cuda();
            qzeros_params_[kv.first] = tensor_copy;
        }
    }
    compute_streams_.fill(nullptr);
    fa_done_events_.fill(nullptr);
    graph_runtime_ = std::make_unique<CudaGraphRuntime<T>>();
    decode_workspace_ = std::make_unique<CudaWorkspaceArena>();

    device_ = Device::CUDA;
    use_cuda_graph_ = use_cuda_graph_ && quant_type_ == 0;
    sample_mode_ = SampleMode::GPU;

    operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
    cpu_operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CPU);
    initialize_cuda_runtime();
    initialize_decode_workspace();
}

template <typename T>
QwenModel<T>::~QwenModel() {
    cleanup_cuda_runtime();
}

template <typename T>
bool QwenModel<T>::verify_params() const {
    // 禁用
    std::cout << "Not checking parameters" << std::endl;
    return true;
}

template <typename T>
void QwenModel<T>::print_model_info() const {
    std::cout << "QwenModel Info:" << std::endl;
    std::cout << "  Vocab size: " << vocab_size_ << std::endl;
    std::cout << "  Layers: " << n_layers_ << std::endl;
    std::cout << "  Heads: " << n_heads_ << std::endl;
    std::cout << "  KV Heads: " << n_kv_heads_ << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
    std::cout << "  Intermediate size: " << intermediate_size_ << std::endl;
    std::cout << "  Max sequence length: " << max_position_embeddings_ << std::endl;
    std::cout << "  RMS Norm eps: " << rms_norm_eps_ << std::endl;
    std::cout << "  RoPE theta: " << rope_theta_ << std::endl;
    std::cout << "  Head dim: " << head_dim_ << std::endl;
    std::cout << "  Device: " << (device_ == Device::CUDA ? "CUDA" : "CPU") << std::endl;
    std::cout << "  Quantization: " << (quant_type_ == 0 ? "None" : (quant_type_ == 1 ? "AWQ" : "Unknown"))
              << std::endl;
    std::cout << "  CUDA Graph: " << (use_cuda_graph_ ? "Enabled" : "Disabled") << std::endl;
    if (quant_type_ != 0) {
        std::cout << "  Group size: " << group_size_ << std::endl;
        std::cout << "  Quantized weights count: " << qweight_params_.size() << std::endl;
        std::cout << "  Scales count: " << scales_params_.size() << std::endl;
        std::cout << "  Zeros count: " << qzeros_params_.size() << std::endl;
    }
}

template <typename T>
void QwenModel<T>::initialize_cuda_runtime() {
    auto &graph = graph_runtime();
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA devices available: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + std::string(cudaGetErrorString(err)));
    }

    if (operators_) {
        operators_->cuda();
    } else {
        operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
    }

    for (cudaStream_t &stream : compute_streams_) {
        if (!stream) {
            err = cudaStreamCreate(&stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
            }
        }
    }

    for (cudaEvent_t &event : fa_done_events_) {
        if (!event) {
            err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to create CUDA event: " + std::string(cudaGetErrorString(err)));
            }
        }
    }

    if (!graph.graph_stream) {
        err = cudaStreamCreate(&graph.graph_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create graph stream: " + std::string(cudaGetErrorString(err)));
        }
    }

    if (!graph.prep_stream) {
        err = cudaStreamCreate(&graph.prep_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create prep stream: " + std::string(cudaGetErrorString(err)));
        }
    }

}

template <typename T>
WorkspacePlan QwenModel<T>::build_decode_workspace_plan() const {
    constexpr size_t kAlignment = CudaMemoryPool::kAllocationAlignment;
    ExecutionProgram program{
        .values =
            {
                {"residual", decode_tensor_shape("residual"), kAlignment},
                {"hidden_states", decode_tensor_shape("hidden_states"), kAlignment},
                {"q_buf", decode_tensor_shape("q_buf"), kAlignment},
                {"att_heads", decode_tensor_shape("att_heads"), kAlignment},
                {"att_proj", decode_tensor_shape("att_proj"), kAlignment},
                {"gate_buf", decode_tensor_shape("gate_buf"), kAlignment},
                {"up_buf", decode_tensor_shape("up_buf"), kAlignment},
                {"ffn_out", decode_tensor_shape("ffn_out"), kAlignment},
                {"logits", decode_tensor_shape("logits"), kAlignment},
            },
        .nodes =
            {
                {"token_embedding", "token_embedding", {}, {"residual"}},
                {"attn_input_norm", "attn_input_norm", {"residual"}, {"hidden_states"}},
                {"q_proj", "q_proj", {"hidden_states"}, {"q_buf"}},
                {"q_rope", "q_rope", {"q_buf"}, {"q_buf"}},
                {"decode_attention", "decode_attention", {"q_buf"}, {"att_heads"}},
                {"o_proj", "o_proj", {"att_heads"}, {"att_proj"}},
                {"attn_residual", "attn_residual", {"residual", "att_proj"}, {"hidden_states", "residual"}},
                {"ffn_up", "ffn_up", {"hidden_states"}, {"gate_buf", "up_buf"}},
                {"ffn_act", "ffn_act", {"gate_buf", "up_buf"}, {"gate_buf"}},
                {"down_proj", "down_proj", {"gate_buf"}, {"ffn_out"}},
                {"ffn_residual", "ffn_residual", {"residual", "ffn_out"}, {"hidden_states", "residual"}},
                {"lm_head", "lm_head", {"hidden_states"}, {"logits"}},
            },
    };
    return build_workspace_plan_from_execution_program<T>(program);
}

template <typename T>
WorkspacePlan QwenModel<T>::build_prefill_workspace_plan(size_t seq_len) const {
    constexpr size_t kAlignment = CudaMemoryPool::kAllocationAlignment;
    ExecutionProgram program{
        .values =
            {
                {"residual", {seq_len, hidden_size_}, kAlignment},
                {"hidden_states", {seq_len, hidden_size_}, kAlignment},
                {"q_buf", {seq_len, n_heads_ * head_dim_}, kAlignment},
                {"k_buf", {seq_len, n_kv_heads_ * head_dim_}, kAlignment},
                {"v_buf", {seq_len, n_kv_heads_ * head_dim_}, kAlignment},
                {"att_heads", {seq_len, n_heads_, head_dim_}, kAlignment},
                {"att_proj", {seq_len, hidden_size_}, kAlignment},
                {"gate_buf", {seq_len, intermediate_size_}, kAlignment},
                {"up_buf", {seq_len, intermediate_size_}, kAlignment},
                {"gate_buf_silu", {seq_len, intermediate_size_}, kAlignment},
                {"ffn_out", {seq_len, hidden_size_}, kAlignment},
                {"logits", {seq_len, vocab_size_}, kAlignment},
            },
        .nodes =
            {
                {"token_embedding", "token_embedding", {}, {"residual"}},
                {"attn_input_norm", "attn_input_norm", {"residual"}, {"hidden_states"}},
                {"qkv_proj", "qkv_proj", {"hidden_states"}, {"q_buf", "k_buf", "v_buf"}},
                {"q_rope", "q_rope", {"q_buf"}, {"q_buf"}},
                {"k_rope", "k_rope", {"k_buf"}, {"k_buf"}},
                {"kv_cache_write", "kv_cache_write", {"k_buf", "v_buf"}, {}},
                {"prefill_attention", "prefill_attention", {"q_buf", "k_buf", "v_buf"}, {"att_heads"}},
                {"o_proj", "o_proj", {"att_heads"}, {"att_proj"}},
                {"attn_residual", "attn_residual", {"residual", "att_proj"}, {"hidden_states", "residual"}},
            },
    };
    if (quant_type_ == 0) {
        program.values.push_back({"merged_mlp_result", {seq_len, 2 * intermediate_size_}, kAlignment});
        program.nodes.push_back(
            {"merged_mlp", "merged_mlp", {"hidden_states"}, {"merged_mlp_result"}});
        program.nodes.push_back(
            {"split_mlp", "split_mlp", {"merged_mlp_result"}, {"gate_buf", "up_buf"}});
    } else {
        program.nodes.push_back(
            {"ffn_up", "ffn_up", {"hidden_states"}, {"gate_buf", "up_buf"}});
    }
    program.nodes.push_back({"ffn_act", "ffn_act", {"gate_buf", "up_buf"}, {"gate_buf_silu"}});
    program.nodes.push_back({"down_proj", "down_proj", {"gate_buf_silu"}, {"ffn_out"}});
    program.nodes.push_back(
        {"ffn_residual", "ffn_residual", {"residual", "ffn_out"}, {"hidden_states", "residual"}});
    program.nodes.push_back({"lm_head", "lm_head", {"hidden_states"}, {"logits"}});
    return build_workspace_plan_from_execution_program<T>(program);
}

template <typename T>
size_t QwenModel<T>::estimate_prefill_workspace_bytes(size_t seq_len) const {
    if (seq_len == 0) {
        return 0;
    }
    const auto plan = build_prefill_workspace_plan(seq_len);
    return plan.total_bytes();
}

template <typename T>
void QwenModel<T>::initialize_decode_workspace() {
    decode_workspace_plan_ = std::make_unique<WorkspacePlan>(build_decode_workspace_plan());
    if (!decode_workspace_ || !decode_workspace_plan_ || decode_workspace_plan_->empty()) {
        return;
    }
    decode_workspace_->reserve_for_plan(*decode_workspace_plan_);
}

template <typename T>
Tensor<T> QwenModel<T>::decode_workspace_tensor(const std::string &name, const std::vector<size_t> &shape) const {
    if (!decode_workspace_ || !decode_workspace_plan_) {
        throw std::runtime_error("Qwen decode workspace is not initialized");
    }
    const auto &allocation = decode_workspace_plan_->at(name);
    const size_t requested_bytes =
        Tensor<T>::from_external_buffer(decode_workspace_->template ptr_at<T>(allocation.offset), shape, Device::CUDA)
            .nbytes();
    if (requested_bytes > allocation.bytes) {
        throw std::runtime_error("Qwen decode workspace allocation too small for " + name);
    }
    return Tensor<T>::from_external_buffer(decode_workspace_->template ptr_at<T>(allocation.offset), shape,
                                           Device::CUDA);
}

template <typename T>
std::vector<size_t> QwenModel<T>::decode_tensor_shape(const std::string &name) const {
    if (name == "residual" || name == "hidden_states" || name == "att_proj" || name == "ffn_out") {
        return {1, hidden_size_};
    }
    if (name == "q_buf") {
        return {1, n_heads_ * head_dim_};
    }
    if (name == "att_heads") {
        return {n_heads_, head_dim_};
    }
    if (name == "gate_buf" || name == "up_buf") {
        return {1, intermediate_size_};
    }
    if (name == "logits") {
        return {1, vocab_size_};
    }
    if (name == "k_buf" || name == "v_buf") {
        return {1, n_kv_heads_ * head_dim_};
    }
    if (name == "fa_output") {
        return {n_heads_, head_dim_ + 2};
    }
    throw std::runtime_error("Unknown Qwen decode tensor shape request: " + name);
}

template <typename T>
std::string QwenModel<T>::graph_tensor_tag(const std::string &name) const {
    return "graph_" + name;
}

template <typename T>
std::string QwenModel<T>::graph_layer_tensor_tag(const std::string &name, size_t layer) const {
    return graph_tensor_tag(name) + "_" + std::to_string(layer);
}

template <typename T>
void QwenModel<T>::cleanup_cuda_runtime() {
    auto &graph = graph_runtime();
    cleanup_graph_fixed_memory();
    graph.release_graph_objects();
    graph.release_streams();

    for (cudaStream_t &stream : compute_streams_) {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    for (cudaEvent_t &event : fa_done_events_) {
        if (event) {
            cudaEventDestroy(event);
            event = nullptr;
        }
    }
    graph.release_pingpong();
    graph.reset_state();
}

template <typename T>
const Tensor<T> *QwenModel<T>::find_optional_param(const std::string &key) const {
    auto it = params_.find(key);
    return it == params_.end() ? nullptr : &it->second;
}

template <typename T>
void QwenModel<T>::add_residual_and_norm(Tensor<T> *hidden_states, Tensor<T> *residual, Tensor<T> *update,
                                         Tensor<T> *norm_weight, cudaStream_t stream) {
    operators_->add_rms(hidden_states, residual, update, norm_weight, rms_norm_eps_, stream);
}

// -------------------------------
// forward_cuda: Qwen2 模型的单个 token CUDA 前向传播
// -------------------------------

template <typename T>
Tensor<T> QwenModel<T>::forward_cuda(const Tensor<uint32_t> *input, KVCache<T> *kv_cache,
                                     const std::string &save_prefix) {
    // 确保输入在 CUDA 上
    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }

    // 获取输入信息，前向传播时序列长度固定为1
    const size_t seq_len = 1;

    // 计算起始KV缓存位置
    size_t offset = 0;
    if (kv_cache) {
        if (kv_cache->device() != Device::CUDA) {
            throw std::runtime_error("KVCache must be on CUDA device");
        }
        offset = kv_cache->size() - seq_len;
    }

    // 创建residual和hidden_states张量(没有batch维度)，使用标签固定内存
    Tensor<T> residual = decode_workspace_tensor("residual", decode_tensor_shape("residual"));
    Tensor<T> hidden_states = decode_workspace_tensor("hidden_states", decode_tensor_shape("hidden_states"));

    //   // 保存输入token（调试用）
    //   std::cout << "forward_cuda: save_prefix = '" << save_prefix << "'"
    //             << std::endl;
    //   if (!save_prefix.empty()) {
    //     std::cout << "保存输入token到: " << save_prefix +
    //     "_cuda_input_token.bin"
    //               << std::endl;
    //     save_uint32_tensor_to_binary(*input, save_prefix +
    //     "_cuda_input_token.bin");

    //     // 调试：打印输入token值和embedding权重信息
    //     std::vector<uint32_t> host_input(input->numel());
    //     cudaMemcpy(host_input.data(), input->data_ptr(),
    //                input->numel() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //     std::cout << "CUDA推理输入token: " << host_input[0] << std::endl;

    //     // 打印embedding权重的地址和前几个值
    //     auto &embedding_weight = params_.at("token_embeddings.weight");
    //     std::cout << "CUDA推理embedding权重地址: " <<
    //     embedding_weight.data_ptr()
    //               << std::endl;
    //     std::cout << "CUDA推理embedding权重形状: [" <<
    //     embedding_weight.sizes()[0]
    //               << ", " << embedding_weight.sizes()[1] << "]" << std::endl;

    //     // 保存当前token对应的embedding向量（用于对比）
    //     uint32_t token_id = host_input[0];
    //     if (token_id < embedding_weight.sizes()[0]) {
    //       // 创建一个临时张量来保存这个token的embedding
    //       Tensor<T> token_embedding(
    //           {1, static_cast<size_t>(embedding_weight.sizes()[1])},
    //           Device::CUDA, false);
    //       size_t offset = token_id * embedding_weight.sizes()[1];
    //       cudaMemcpy(token_embedding.data_ptr(),
    //                  static_cast<const T *>(embedding_weight.data_ptr()) +
    //                  offset, embedding_weight.sizes()[1] * sizeof(T),
    //                  cudaMemcpyDeviceToDevice);
    //       save_tensor_to_binary(token_embedding, save_prefix + "_cuda_token_" +
    //                                                  std::to_string(token_id) +
    //                                                  "_embedding.bin");
    //       std::cout << "保存了token " << token_id << " 的embedding向量"
    //                 << std::endl;
    //     }
    //   } else {
    //     std::cout << "save_prefix为空，跳过保存输入token" << std::endl;
    //   }

    // Token嵌入 (从embedding_table中获取token嵌入)
    operators_->gather(&residual, input, &params_.at("token_embeddings.weight"));

    // 保存初始嵌入
    if (!save_prefix.empty()) {
        save_tensor_to_binary(residual, save_prefix + "_cuda_embedding.bin");
    }

    // cudaStreamSynchronize(compute_streams_[3]);
    // cudaStreamSynchronize(compute_streams_[4]);
    // 主循环：遍历所有Transformer层
    std::string l = "layers." + std::to_string(0) + ".";
    auto &attention_norm_weight = params_.at(l + "input_layernorm.weight");
    // 使用新的算子抽象层
    operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight, rms_norm_eps_);

    if (!save_prefix.empty()) {
        save_tensor_to_binary(hidden_states, save_prefix + "_cuda_layer_0_input_norm.bin");
    }
    for (size_t i = 0; i < n_layers_; i++) {
        std::string layer_prefix = "layers." + std::to_string(i) + ".";

        // 2. Self-Attention
        Tensor<T> q_buf = decode_workspace_tensor("q_buf", decode_tensor_shape("q_buf"));
        Tensor<T> &k_slice = kv_cache->k_cache(i, offset);
        Tensor<T> &v_slice = kv_cache->v_cache(i, offset);

        // 获取偏置项（如果存在）
        const Tensor<T> *q_bias = nullptr;
        const Tensor<T> *k_bias = nullptr;
        const Tensor<T> *v_bias = nullptr;
        const Tensor<T> *o_bias = nullptr;

        try {
            q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
        } catch (const std::out_of_range &) {
        }

        auto q_weight = get_weight(layer_prefix + "self_attn.q_proj");
        auto k_weight = get_weight(layer_prefix + "self_attn.k_proj");
        auto v_weight = get_weight(layer_prefix + "self_attn.v_proj");

        operators_->matmul(&q_buf, &hidden_states, q_weight, q_bias);
        operators_->matmul(&k_slice, &hidden_states, k_weight, k_bias);
        operators_->matmul(&v_slice, &hidden_states, v_weight, v_bias);

        // // 保存QKV投影结果
        // if (!save_prefix.empty()) {
        //   save_tensor_to_binary(q_buf, save_prefix + "_cuda_layer_" +
        //                                    std::to_string(i) + "_q_proj.bin");
        //   save_tensor_to_binary(k_slice, save_prefix + "_cuda_layer_" +
        //                                      std::to_string(i) + "_k_proj.bin");
        //   save_tensor_to_binary(v_slice, save_prefix + "_cuda_layer_" +
        //                                      std::to_string(i) + "_v_proj.bin");
        // }

        // 重塑张量，准备应用RoPE
        Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
        Tensor<T> k_buf_view = k_slice.view({seq_len, n_kv_heads_, head_dim_});
        Tensor<T> v_buf_view = v_slice.view({seq_len, n_kv_heads_, head_dim_});

        // for (size_t j = 0; j < seq_len; j++) {
        //   // 获取对应的 k 和 v slice

        //   Tensor<T>& k_slice = kv_cache->k_cache(i, offset + j);
        //   Tensor<T>& v_slice = kv_cache->v_cache(i, offset + j);
        //   // debugPrintTensor(k_slice, "k_slice");
        //   // debugPrintTensor(v_slice, "v_slice");
        //   // 将数据从 k_buf_view 和 v_buf_view 拷贝到对应 slice 的内存中
        //   cudaMemcpy(k_slice.data_ptr(), k_buf_view.data_ptr() + j * row_size,
        //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
        //   cudaMemcpy(v_slice.data_ptr(), v_buf_view.data_ptr() + j * row_size,
        //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
        //   // debugPrintTensor(k_slice, "k_slice");
        //   // debugPrintTensor(v_slice, "v_slice");
        // }

        // 应用旋转位置编码 (RoPE)
        // 使用新的算子抽象层（二重指针版本）
        size_t offset_q = offset;
        size_t offset_k = offset;
        operators_->rope(&q_buf_view, offset_q, rope_theta_, nullptr);
        operators_->rope(&k_buf_view, offset_k, rope_theta_, nullptr);

        // // 保存RoPE后的结果
        // if (!save_prefix.empty()) {
        //   save_tensor_to_binary(q_buf_view, save_prefix + "_cuda_layer_" +
        //                                         std::to_string(i) +
        //                                         "_q_rope.bin");
        //   save_tensor_to_binary(k_buf_view, save_prefix + "_cuda_layer_" +
        //                                         std::to_string(i) +
        //                                         "_k_rope.bin");
        // }

        // 更新KV缓存
        // size_t row_size = n_kv_heads_ * head_dim_;

        // for (size_t j = 0; j < seq_len; j++) {
        //   // 获取对应的 k 和 v slice
        //   Tensor<T>& k_slice = kv_cache->k_cache(i, offset + j);
        //   Tensor<T>& v_slice = kv_cache->v_cache(i, offset + j);

        //   // 异步拷贝：使用 cudaMemcpyAsync 替换同步版本
        //   cudaError_t err1 = cudaMemcpyAsync(
        //       k_slice.data_ptr(), k_buf_view.data_ptr() + j * row_size,
        //       row_size * sizeof(T), cudaMemcpyDeviceToDevice,
        //       compute_streams_[3]);
        //   cudaError_t err2 = cudaMemcpyAsync(
        //       v_slice.data_ptr(), v_buf_view.data_ptr() + j * row_size,
        //       row_size * sizeof(T), cudaMemcpyDeviceToDevice,
        //       compute_streams_[4]);
        // }

        // 准备计算自注意力
        Tensor<T> Q_3d = q_buf_view;
        Tensor<T> total_K, total_V;
        size_t total_seq_len = seq_len;

        // 如果有缓存，拼接当前和缓存的K,V
        if (offset != 0) {
            size_t cached_len = offset;
            total_seq_len = cached_len + seq_len;
            // total_K =
            //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);
            // // total_KX =
            // //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_},
            // Device::CUDA); total_V =
            //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);

            // // 拷贝缓存的K,V
            // for (size_t pos = 0; pos < cached_len; pos++) {
            //   Tensor<T>& cached_k = kv_cache->k_cache(i, pos);
            //   Tensor<T>& cached_v = kv_cache->v_cache(i, pos);
            //   cudaMemcpy(total_K.data_ptr() + pos * row_size, cached_k.data_ptr(),
            //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
            //   cudaMemcpy(total_V.data_ptr() + pos * row_size, cached_v.data_ptr(),
            //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
            //   // debugPrintTensor(cached_k, "cached_k");
            //   // debugPrintTensor(cached_v, "cached_v");
            // }
            // // 拷贝当前的K,V

            // cudaMemcpy(total_K.data_ptr() + cached_len * row_size,
            //            k_buf_view.data_ptr(), seq_len * row_size * sizeof(T),
            //            cudaMemcpyDeviceToDevice);
            // cudaMemcpy(total_V.data_ptr() + cached_len * row_size,
            //            v_buf_view.data_ptr(), seq_len * row_size * sizeof(T),
            //            cudaMemcpyDeviceToDevice);

            auto [total_K1, total_V1] = kv_cache->get_contiguous_tensor(i);
            // total_KX.view({total_seq_len, n_kv_heads_, head_dim_});
            // debugPrintTensor(total_KX, "total_KX");
            // debugPrintTensor(total_K, "total_K");

            total_K = total_K1.view({total_seq_len, n_kv_heads_, head_dim_});
            total_V = total_V1.view({total_seq_len, n_kv_heads_, head_dim_});

        } else {
            total_K = k_buf_view;
            total_V = v_buf_view;
        }

        Tensor<T> att_heads = decode_workspace_tensor("att_heads", decode_tensor_shape("att_heads"));

        // Tensor<T> att_heads_1({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

        // Tensor<T> att_heads_2({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

        // Tensor<T> att_heads_3({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

        // // Tensor<T> att_heads_4({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

        // // Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA);
        // size_t seq_divide = total_seq_len / 3;
        // int ratio = n_heads_ / n_kv_heads_;
        // // for (int j = 0; j < 3; ++j) {
        // //   cudaStreamSynchronize(compute_streams_[j]);
        // // }

        // 保存完整的KV
        // if (!save_prefix.empty()) {
        //   save_tensor_to_binary(total_K, save_prefix + "_cuda_layer_" +
        //                                      std::to_string(i) + "_total_k.bin");
        //   save_tensor_to_binary(total_V, save_prefix + "_cuda_layer_" +
        //                                      std::to_string(i) + "_total_v.bin");
        // }

        // 使用新的动态分支数量的flash attention包装函数
        operators_->dynamic_flash_attention(Q_3d, total_K, total_V, att_heads, n_kv_heads_);

        // 保存attention输出
        // if (!save_prefix.empty()) {
        //   save_tensor_to_binary(
        //       att_heads,
        //       save_prefix + "_cuda_layer_" + std::to_string(i) +
        //       "_att_heads.bin");
        // }
        Tensor<T> att_heads_reshaped = att_heads.view({1, n_heads_ * head_dim_});
        Tensor<T> att_proj = decode_workspace_tensor("att_proj", decode_tensor_shape("att_proj"));

        auto o_weight = get_weight(layer_prefix + "self_attn.o_proj");

        // 执行矩阵乘法（统一接口）
        operators_->matmul(&att_proj, &att_heads_reshaped, o_weight, o_bias);

        // 保存attention投影
        if (!save_prefix.empty()) {
            save_tensor_to_binary(att_proj, save_prefix + "_cuda_layer_" + std::to_string(i) + "_att_proj.bin");
        }

        // 残差连接
        // cudaDeviceSynchronize();
        auto &ffn_norm_weight = params_.at(layer_prefix + "post_attention_layernorm.weight");

        add_residual_and_norm(&hidden_states, &residual, &att_proj, &ffn_norm_weight);

        // 保存post attention状态
        if (!save_prefix.empty()) {
            save_tensor_to_binary(residual,
                                  save_prefix + "_cuda_layer_" + std::to_string(i) + "_post_att_residual.bin");
            save_tensor_to_binary(hidden_states,
                                  save_prefix + "_cuda_layer_" + std::to_string(i) + "_post_att_norm.bin");
        }
        // debugPrintTensor(hidden_states, "hidden_states-after");

        // 获取偏置项（如果存在）
        const Tensor<T> *gate_bias = nullptr;
        const Tensor<T> *up_bias = nullptr;
        const Tensor<T> *down_bias = nullptr;

        // 预先分配输出张量
        size_t intermediate_size = intermediate_size_;
        if (quant_type_ == 0) {
            // 非量化版本，从权重获取维度
            auto &gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
            intermediate_size = gate_weight.sizes()[1];
        }

        Tensor<T> gate_buf = decode_workspace_tensor("gate_buf", {seq_len, intermediate_size});
        Tensor<T> up_buf = decode_workspace_tensor("up_buf", {seq_len, intermediate_size});

        auto gate_weight = get_weight(layer_prefix + "mlp.gate_proj");
        auto up_weight = get_weight(layer_prefix + "mlp.up_proj");

        // 执行矩阵乘法
        operators_->matmul(&gate_buf, &hidden_states, gate_weight, gate_bias);
        operators_->matmul(&up_buf, &hidden_states, up_weight, up_bias);

        // // 保存MLP投影
        // if (!save_prefix.empty()) {
        //   save_tensor_to_binary(gate_buf, save_prefix + "_cuda_layer_" +
        //                                       std::to_string(i) +
        //                                       "_gate_proj.bin");
        //   save_tensor_to_binary(up_buf, save_prefix + "_cuda_layer_" +
        //                                     std::to_string(i) + "_up_proj.bin");
        // }

        // 使用新的算子抽象层
        operators_->silu(&gate_buf, &gate_buf);               // SiLU激活
        operators_->multiply(&gate_buf, &gate_buf, &up_buf);  // 逐元素相乘

        // 保存激活后的结果
        if (!save_prefix.empty()) {
            save_tensor_to_binary(gate_buf, save_prefix + "_cuda_layer_" + std::to_string(i) + "_silu_mul.bin");
        }

        // 投影回原始维度
        Tensor<T> ffn_out = decode_workspace_tensor("ffn_out", decode_tensor_shape("ffn_out"));

        // 改为:
        // 获取权重（自动处理量化与非量化情况）
        auto down_weight = get_weight(layer_prefix + "mlp.down_proj");

        // 执行矩阵乘法（统一接口）
        operators_->matmul(&ffn_out, &gate_buf, down_weight, down_bias);

        // 保存down投影
        if (!save_prefix.empty()) {
            save_tensor_to_binary(ffn_out, save_prefix + "_cuda_layer_" + std::to_string(i) + "_down_proj.bin");
        }

        if (i == n_layers_ - 1) {
            // 最后一层的残差连接，直接与最终的RMS norm合并
            auto &norm_weight = params_.at("norm.weight");
            add_residual_and_norm(&hidden_states, &residual, &ffn_out, &norm_weight);
        } else {
            std::string lx = "layers." + std::to_string(i + 1) + ".";
            auto &attention_norm_weight = params_.at(lx + "input_layernorm.weight");
            add_residual_and_norm(&hidden_states, &residual, &ffn_out, &attention_norm_weight);
        }

        // 保存层结束后的状态
        // if (!save_prefix.empty()) {
        //   save_tensor_to_binary(residual, save_prefix + "_cuda_layer_" +
        //                                       std::to_string(i) +
        //                                       "_final_residual.bin");
        //   if (i < n_layers_ - 1) {
        //     save_tensor_to_binary(hidden_states, save_prefix + "_cuda_layer_" +
        //                                              std::to_string(i) +
        //                                              "_final_hidden.bin");
        //   }
        // }
    }

    // 最终的LayerNorm已经在最后一层的循环中合并处理了
    Tensor<T> final_h = hidden_states;

    // 保存最终norm
    if (!save_prefix.empty()) {
        save_tensor_to_binary(final_h, save_prefix + "_cuda_final_norm.bin");
    }

    // LM head投影到词汇表大小
    auto lm_head_weight = get_weight("lm_head");

    const Tensor<T> *lm_head_bias = nullptr;
    // try {
    //   lm_head_bias = &params_.at("lm_head_bias");
    //   std::cout << "Found lm_head_bias" << std::endl;
    // } catch (const std::out_of_range&) {
    // }

    Tensor<T> logits = decode_workspace_tensor("logits", decode_tensor_shape("logits"));
    operators_->matmul(&logits, &final_h, lm_head_weight, lm_head_bias);

    // 保存最终logits
    if (!save_prefix.empty()) {
        save_tensor_to_binary(logits, save_prefix + "_cuda_final_logits.bin");
    }

    // 返回最后一个token的logits

    return logits;
}

// -------------------------------
// prefill_cuda: Qwen2 模型的序列预填充 CUDA 实现
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::prefill_cuda(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    auto &graph = graph_runtime();
    // 确保输入在 CUDA 上
    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }

    // 获取输入信息
    const size_t seq_len = input->sizes()[0];

    // 计算起始KV缓存位置
    size_t offset = 0;
    if (kv_cache) {
        if (kv_cache->device() != Device::CUDA) {
            throw std::runtime_error("KVCache must be on CUDA device");
        }
        offset = kv_cache->size() - seq_len;
    }

    // 创建residual和hidden_states张量，在prefill阶段自动使用prefill buffer
    Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA);
    Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA);

    // Token嵌入 (从embedding_table中获取token嵌入)
    operators_->gather(&residual, input, &params_.at("token_embeddings.weight"));
    std::string l = "layers." + std::to_string(0) + ".";
    auto &attention_norm_weight = params_.at(l + "input_layernorm.weight");
    // 使用新的算子抽象层
    operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight, rms_norm_eps_);

    // 提前准备RoPE offset，复用forward_for_graph的机制
    if (graph.d_rope_offset == nullptr) {
        cudaMalloc(&graph.d_rope_offset, sizeof(size_t) * 2);
    }
    cudaMemcpy(graph.d_rope_offset, &offset, sizeof(size_t), cudaMemcpyHostToDevice);

    // 主循环：遍历所有Transformer层
    for (size_t i = 0; i < n_layers_; i++) {
        std::string layer_prefix = "layers." + std::to_string(i) + ".";
        // auto &attention_norm_weight = params_.at(layer_prefix + "input_layernorm.weight");
        // operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight, rms_norm_eps_);

        const Tensor<T> *q_bias = nullptr;
        const Tensor<T> *k_bias = nullptr;
        const Tensor<T> *v_bias = nullptr;
        const Tensor<T> *o_bias = nullptr;
        try {
            q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
        } catch (const std::out_of_range &) {
        }

        Tensor<T> q_buf, k_buf, v_buf;

        // QKV融合优化：对于非量化模型，检查是否有合并的QKV权重
        std::string merged_qkv_key = "merged_qkv_" + std::to_string(i);
        if (quant_type_ == 0 && params_.find(merged_qkv_key) != params_.end()) {
            // 使用合并的QKV权重进行单次矩阵乘法
            auto merged_qkv_weight = params_.at(merged_qkv_key);

            // 计算合并后的输出维度
            size_t q_dim = n_heads_ * head_dim_;
            size_t k_dim = n_kv_heads_ * head_dim_;
            size_t v_dim = n_kv_heads_ * head_dim_;
            size_t total_dim = q_dim + k_dim + v_dim;

            // 合并的QKV投影：一次矩阵乘法得到Q、K和V
            Tensor<T> merged_qkv_result({seq_len, total_dim}, Device::CUDA);

            // 处理bias（如果存在）
            const Tensor<T> *merged_qkv_bias = nullptr;
            try {
                merged_qkv_bias = &params_.at("merged_qkv_bias_" + std::to_string(i));
            } catch (const std::out_of_range &) {
                // 如果没有bias则为空
            }

            // 执行合并的矩阵乘法
            operators_->matmul(&merged_qkv_result, &hidden_states, get_weight(merged_qkv_key), merged_qkv_bias);

            // 从合并结果中拆分Q、K、V（使用slice，保持非连续stride）
            q_buf = merged_qkv_result.slice({0, 0}, {seq_len, q_dim});
            k_buf = merged_qkv_result.slice({0, q_dim}, {seq_len, q_dim + k_dim});
            v_buf = merged_qkv_result.slice({0, q_dim + k_dim}, {seq_len, total_dim});

        } else {
            // 量化模式或没有合并权重：分别计算Q、K、V
            q_buf = Tensor<T>({seq_len, n_heads_ * head_dim_}, Device::CUDA);
            k_buf = Tensor<T>({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);
            v_buf = Tensor<T>({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);

            auto wq_ptr = get_weight(layer_prefix + "self_attn.q_proj");
            auto wk_ptr = get_weight(layer_prefix + "self_attn.k_proj");
            auto wv_ptr = get_weight(layer_prefix + "self_attn.v_proj");

            operators_->matmul(&q_buf, &hidden_states, wq_ptr, q_bias);
            operators_->matmul(&k_buf, &hidden_states, wk_ptr, k_bias);
            operators_->matmul(&v_buf, &hidden_states, wv_ptr, v_bias);
        }

        // 调整张量形状以进行后续操作（保持非连续stride）
        Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
        Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});
        Tensor<T> v_buf_view = v_buf.view({seq_len, n_kv_heads_, head_dim_});

        operators_->rope(&q_buf_view, offset, rope_theta_);
        operators_->rope(&k_buf_view, offset, rope_theta_);
        decoder::write_kv_cache(device_, kv_cache, i, offset, n_kv_heads_, head_dim_, k_buf_view, v_buf_view);

        // 重新格式化Q用于注意力计算
        Tensor<T> Q_3d = q_buf_view;

        // 准备K和V张量用于注意力计算
        Tensor<T> total_K, total_V;
        size_t total_seq_len = 0;

        total_seq_len = offset + seq_len;

        // total_K =
        //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);
        // total_V =
        //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);

        // // 拷贝缓存的K,V
        // for (size_t pos = 0; pos < offset; pos++) {
        //   const auto& cached_k = kv_cache->k_cache(i, pos);
        //   const auto& cached_v = kv_cache->v_cache(i, pos);

        //   cudaMemcpy(total_K.data_ptr() + pos * n_kv_heads_ * head_dim_,
        //              cached_k.data_ptr(), n_kv_heads_ * head_dim_ * sizeof(T),
        //              cudaMemcpyDeviceToDevice);

        //   cudaMemcpy(total_V.data_ptr() + pos * n_kv_heads_ * head_dim_,
        //              cached_v.data_ptr(), n_kv_heads_ * head_dim_ * sizeof(T),
        //              cudaMemcpyDeviceToDevice);
        // }
        // // 拷贝当前的K, V

        // cudaMemcpy(total_K.data_ptr() + offset * n_kv_heads_ * head_dim_,
        //            k_buf_view.data_ptr(),
        //            seq_len * n_kv_heads_ * head_dim_ * sizeof(T),
        //            cudaMemcpyDeviceToDevice);

        // cudaMemcpy(total_V.data_ptr() + offset * n_kv_heads_ * head_dim_,
        //            v_buf_view.data_ptr(),
        //            seq_len * n_kv_heads_ * head_dim_ * sizeof(T),
        //            cudaMemcpyDeviceToDevice);
        auto [total_K1, total_V1] = kv_cache->get_contiguous_tensor(i);
        // total_KX.view({total_seq_len, n_kv_heads_, head_dim_});
        // debugPrintTensor(total_KX, "total_KX");
        // debugPrintTensor(total_K, "total_K");

        total_K = total_K1.view({total_seq_len, n_kv_heads_, head_dim_});
        // debugPrintTensor(total_K, "total_K");
        total_V = total_V1.view({total_seq_len, n_kv_heads_, head_dim_});

        Tensor<T> att_heads({seq_len, n_heads_, head_dim_}, Device::CUDA);

        operators_->flash_attention_prefill(Q_3d, total_K, total_V, att_heads, n_heads_, n_kv_heads_, head_dim_,
                                            seq_len, total_seq_len, offset);

        // 将注意力输出投影回原始维度
        Tensor<T> att_proj({seq_len, hidden_size_}, Device::CUDA);

        auto o_weight = get_weight(layer_prefix + "self_attn.o_proj");
        operators_->matmul(&att_proj, &att_heads.view({seq_len, n_heads_ * head_dim_}), o_weight, o_bias);


        auto &ffn_norm_weight = params_.at(layer_prefix + "post_attention_layernorm.weight");
        add_residual_and_norm(&hidden_states, &residual, &att_proj, &ffn_norm_weight);

        const Tensor<T> *gate_bias = nullptr;
        const Tensor<T> *up_bias = nullptr;
        const Tensor<T> *down_bias = nullptr;

        try {
            gate_bias = &params_.at(layer_prefix + "mlp.gate_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            up_bias = &params_.at(layer_prefix + "mlp.up_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            down_bias = &params_.at(layer_prefix + "mlp.down_proj.bias");
        } catch (const std::out_of_range &) {
        }

        Tensor<T> gate_buf;
        Tensor<T> up_buf;
        Tensor<T> merged_mlp_result;

        if (quant_type_ == 1) {
            gate_buf = Tensor<T>({seq_len, intermediate_size_}, Device::CUDA);
            up_buf = Tensor<T>({seq_len, intermediate_size_}, Device::CUDA);
            auto gate_weight = get_weight(layer_prefix + "mlp.gate_proj");
            auto up_weight = get_weight(layer_prefix + "mlp.up_proj");
            operators_->matmul(&gate_buf, &hidden_states, gate_weight, gate_bias);
            operators_->matmul(&up_buf, &hidden_states, up_weight, up_bias);
        } else {
            // 实际上，0也可以走上面的封装，但这里测试mlp并行
            // 没有bias，处理起来比较方便
            // 当然，只支持seqlen为1
            merged_mlp_result = Tensor<T>({seq_len, 2 * intermediate_size_}, Device::CUDA);
            operators_->matmul(&merged_mlp_result, &hidden_states, get_weight("merged_mlp_" + std::to_string(i)));
            gate_buf = merged_mlp_result.slice({0, 0}, {seq_len, intermediate_size_});
            up_buf = merged_mlp_result.slice({0, intermediate_size_}, {seq_len, 2 * intermediate_size_});
        }
        Tensor<T> gate_buf_silu({seq_len, intermediate_size_}, Device::CUDA);
        operators_->silu_multiply(&gate_buf_silu, &gate_buf, &up_buf);

        // 投影回原始维度
        Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA);

        // 获取down_proj权重（自动处理量化与非量化情况）
        auto down_weight = get_weight(layer_prefix + "mlp.down_proj");

        // 执行矩阵乘法（内部自动选择合适的实现）
        operators_->matmul(&ffn_out, &gate_buf_silu, down_weight, down_bias);

        if (i == n_layers_ - 1) {
            // 最后一层的残差连接，直接与最终的RMS norm合并
            auto &norm_weight = params_.at("norm.weight");
            add_residual_and_norm(&hidden_states, &residual, &ffn_out, &norm_weight);
        } else {
            std::string lx = "layers." + std::to_string(i + 1) + ".";
            auto &attention_norm_weight = params_.at(lx + "input_layernorm.weight");
            add_residual_and_norm(&hidden_states, &residual, &ffn_out, &attention_norm_weight);
        }
    }

    // 最终的LayerNorm已经在最后一层的循环中合并处理了
    Tensor<T> final_h = hidden_states;

    auto lm_head_weight = get_weight("lm_head");

    const Tensor<T> *lm_head_bias = nullptr;

    Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
    operators_->matmul(&logits, &final_h, lm_head_weight, lm_head_bias);

    return logits;
}

template <typename T>
Tensor<T> QwenModel<T>::forward_generic(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    if (input->device() != device_) {
        throw std::runtime_error("Input tensor device does not match QwenModel device");
    }
    if (quant_type_ != 0) {
        throw std::runtime_error("Generic Qwen forward only supports non-quantized weights");
    }

    decoder::DecoderGeometry geometry = {n_layers_,         n_heads_,      n_kv_heads_, hidden_size_,
                                         head_dim_,         intermediate_size_, vocab_size_, rms_norm_eps_,
                                         rope_theta_};
    decoder::CommonDecoderWeights<T> common_weights = {
        &params_.at("token_embeddings.weight"),
        &params_.at("norm.weight"),
        get_weight("lm_head"),
        find_optional_param("lm_head.bias"),
    };
    auto resolve_layer = [this](size_t layer_idx) {
        const std::string layer_prefix = "layers." + std::to_string(layer_idx) + ".";
        return decoder::DecoderLayerWeights<T>{
            &params_.at(layer_prefix + "input_layernorm.weight"),
            get_weight(layer_prefix + "self_attn.q_proj"),
            find_optional_param(layer_prefix + "self_attn.q_proj.bias"),
            get_weight(layer_prefix + "self_attn.k_proj"),
            find_optional_param(layer_prefix + "self_attn.k_proj.bias"),
            get_weight(layer_prefix + "self_attn.v_proj"),
            find_optional_param(layer_prefix + "self_attn.v_proj.bias"),
            get_weight(layer_prefix + "self_attn.o_proj"),
            find_optional_param(layer_prefix + "self_attn.o_proj.bias"),
            &params_.at(layer_prefix + "post_attention_layernorm.weight"),
            get_weight(layer_prefix + "mlp.gate_proj"),
            find_optional_param(layer_prefix + "mlp.gate_proj.bias"),
            get_weight(layer_prefix + "mlp.up_proj"),
            find_optional_param(layer_prefix + "mlp.up_proj.bias"),
            get_weight(layer_prefix + "mlp.down_proj"),
            find_optional_param(layer_prefix + "mlp.down_proj.bias"),
        };
    };
    return decoder::run_generic_decoder(input, kv_cache, device_, operators_.get(),
                                        geometry, common_weights, resolve_layer);
}

template <typename T>
Tensor<T> QwenModel<T>::prefill_generic(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    return forward_generic(input, kv_cache);
}

// -------------------------------
// cuda()：将所有参数移到 CUDA，并设置设备
// -------------------------------
template <typename T>
QwenModel<T> &QwenModel<T>::cuda() {
    initialize_cuda_runtime();

    for (auto &kv : params_) {
        if (kv.second.device() != Device::CUDA) {
            kv.second.cuda();
        }
    }
    device_ = Device::CUDA;

    // 更新算子接口
    if (operators_) {
        operators_->cuda();
    } else {
        operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
    }

    sample_mode_ = SampleMode::GPU;
    use_cuda_graph_ = quant_type_ == 0;

    // CUDA图初始化现在延迟到第一次forward调用时进行
    // 这样可以使用真实的KV cache
    if (quant_type_ == 0) {
        std::cout << "预计算RoPE sin/cos值, head_dim: " << head_dim_ << std::endl;
        std::vector<float> freq_(head_dim_ / 2);  // RoPE只需要head_dim/2个频率
        for (int i = 0; i < head_dim_ / 2; ++i) {
            freq_[i] = 1.0f / powf(rope_theta_, (2.0f * i) / static_cast<float>(head_dim_));
        }

        size_t max_seq_len = max_position_embeddings_;
        std::vector<float> sin_cos_cpu(2 * max_seq_len * head_dim_ / 2);  // 存储[pos][freq][sin,cos]格式
        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (int i = 0; i < head_dim_ / 2; ++i) {
                float angle = static_cast<float>(pos) * freq_[i];
                size_t base_idx = pos * head_dim_ + i * 2;
                sin_cos_cpu[base_idx] = sinf(angle);      // sin值
                sin_cos_cpu[base_idx + 1] = cosf(angle);  // cos值
            }
        }

        rope_sin_cos_cache_ = Tensor<float>({max_seq_len, head_dim_}, Device::CUDA, false, "rope_sin_cos_cache");

        cudaError_t err = cudaMemcpy(rope_sin_cos_cache_.data_ptr(), sin_cos_cpu.data(),
                                     sin_cos_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy RoPE sin/cos cache to GPU: " +
                                     std::string(cudaGetErrorString(err)));
        }

        std::cout << "RoPE sin/cos缓存已创建，形状: [" << max_seq_len << ", " << head_dim_ << "]" << std::endl;

        std::vector<float>().swap(freq_);
        std::vector<float>().swap(sin_cos_cpu);
        // 合并Q、K和V权重，通过cudamemcpy
        for (int i = 0; i < n_layers_; i++) {
            std::string layer_prefix = "layers." + std::to_string(i) + ".";
            auto q_weight = params_.at(layer_prefix + "self_attn.q_proj.weight");
            auto k_weight = params_.at(layer_prefix + "self_attn.k_proj.weight");
            auto v_weight = params_.at(layer_prefix + "self_attn.v_proj.weight");

            // 创建合并的QKV权重张量：[hidden_size, n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim]
            size_t total_width = q_weight.sizes()[1] + k_weight.sizes()[1] + v_weight.sizes()[1];
            Tensor<T> merged_qkv_weight({q_weight.sizes()[0], total_width}, Device::CUDA, false,
                                        "merged_qkv_" + std::to_string(i));

            // 按顺序拷贝Q、K、V权重
            cudaMemcpy(merged_qkv_weight.data_ptr(), q_weight.data_ptr(), q_weight.nbytes(), cudaMemcpyDeviceToDevice);
            cudaMemcpy(merged_qkv_weight.data_ptr() + q_weight.numel(), k_weight.data_ptr(), k_weight.nbytes(),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(merged_qkv_weight.data_ptr() + q_weight.numel() + k_weight.numel(), v_weight.data_ptr(),
                       v_weight.nbytes(), cudaMemcpyDeviceToDevice);
            params_["merged_qkv_" + std::to_string(i)] = std::move(merged_qkv_weight);

            // 合并QKV bias（如果存在）
            try {
                auto q_bias = params_.at(layer_prefix + "self_attn.q_proj.bias");
                auto k_bias = params_.at(layer_prefix + "self_attn.k_proj.bias");
                auto v_bias = params_.at(layer_prefix + "self_attn.v_proj.bias");

                size_t total_bias_size = q_bias.sizes()[0] + k_bias.sizes()[0] + v_bias.sizes()[0];
                Tensor<T> merged_qkv_bias({total_bias_size}, Device::CUDA, false,
                                          "merged_qkv_bias_" + std::to_string(i));

                cudaMemcpy(merged_qkv_bias.data_ptr(), q_bias.data_ptr(), q_bias.nbytes(), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged_qkv_bias.data_ptr() + q_bias.numel(), k_bias.data_ptr(), k_bias.nbytes(),
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged_qkv_bias.data_ptr() + q_bias.numel() + k_bias.numel(), v_bias.data_ptr(),
                           v_bias.nbytes(), cudaMemcpyDeviceToDevice);
                params_["merged_qkv_bias_" + std::to_string(i)] = std::move(merged_qkv_bias);
            } catch (const std::out_of_range &) {
                // 如果没有bias则跳过
            }
        }
        std::cout << "合并MLP权重" << std::endl;
        for (int i = 0; i < n_layers_; i++) {
            std::string layer_prefix = "layers." + std::to_string(i) + ".";
            auto gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
            auto up_weight = params_.at(layer_prefix + "mlp.up_proj.weight");

            Tensor<T> merged_weight({gate_weight.sizes()[0], 2 * gate_weight.sizes()[1]}, Device::CUDA, false,
                                    "merged_mlp_" + std::to_string(i));
            cudaMemcpy(merged_weight.data_ptr(), gate_weight.data_ptr(), gate_weight.nbytes(),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(merged_weight.data_ptr() + gate_weight.numel(), up_weight.data_ptr(), up_weight.nbytes(),
                       cudaMemcpyDeviceToDevice);
            params_["merged_mlp_" + std::to_string(i)] = std::move(merged_weight);
        }
    }

    initialize_decode_workspace();

    return *this;
}

// -------------------------------
// cpu()：Qwen 模型仅支持 CUDA，故调用 cpu() 抛出异常
// -------------------------------
template <typename T>
QwenModel<T> &QwenModel<T>::cpu() {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        throw std::runtime_error("Qwen BF16 CPU execution is not supported yet.");
    }
    if (quant_type_ != 0) {
        throw std::runtime_error("Quantized Qwen CPU execution is not supported yet.");
    }

    for (auto &kv : params_) {
        if (kv.second.device() != Device::CPU) {
            kv.second.cpu();
        }
    }

    if (operators_) {
        operators_->cpu();
    } else {
        operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CPU);
    }

    device_ = Device::CPU;
    sample_mode_ = SampleMode::CPU;
    use_cuda_graph_ = false;
    cleanup_cuda_runtime();
    return *this;
}

// -------------------------------
// generate: Token 生成接口，目前作为 stub
// -------------------------------
template <typename T>
std::vector<uint32_t> QwenModel<T>::generate(const std::vector<uint32_t> &input_ids, size_t max_length,
                                             float temperature, float top_p, size_t top_k) {
    // TODO: 实现 Qwen 模型的 token 生成逻辑
    throw std::runtime_error("Token generation not implemented for QwenModel");
    return std::vector<uint32_t>();
}

// -------------------------------
// forward_for_graph: Qwen2 模型的前向传播，用于图优化开发
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::forward_for_graph(const Tensor<uint32_t> *input, KVCache<T> *kv_cache, cudaStream_t stream) {
    auto &graph = graph_runtime();
    // 基本输入检查
    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }

    const size_t seq_len = 1;  // 单token推理

    Tensor<T> residual(decode_tensor_shape("residual"), Device::CUDA, false, graph_tensor_tag("residual"));
    Tensor<T> hidden_states(decode_tensor_shape("hidden_states"), Device::CUDA, false, graph_tensor_tag("hidden_states"));

    std::cout << "gather操作参数地址:" << std::endl;
    std::cout << "  residual地址: " << residual.data_ptr() << std::endl;
    std::cout << "  graph_input_tensor_地址: " << graph.graph_input_tensor.data_ptr() << std::endl;
    std::cout << "  embedding权重地址: " << params_.at("token_embeddings.weight").data_ptr() << std::endl;

    operators_->gather(&residual, &graph.graph_input_tensor, &params_.at("token_embeddings.weight"), stream);
    std::string l = "layers." + std::to_string(0) + ".";
    auto &attention_norm_weight = params_.at(l + "input_layernorm.weight");
    operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight, rms_norm_eps_, stream);

    for (size_t i = 0; i < n_layers_; i++) {
        std::string layer_prefix = "layers." + std::to_string(i) + ".";

        Tensor<T> q_buf(decode_tensor_shape("q_buf"), Device::CUDA, false, graph_layer_tensor_tag("q_buf", i));

        // 适配初版路径
        // Tensor<T> &k_buf = fixed_k_buffers_[i];
        // Tensor<T> &v_buf = fixed_v_buffers_[i];

        // 第二版路径
        Tensor<T> &k_buf = kv_cache->k_cache(0, 0);
        Tensor<T> &v_buf = kv_cache->v_cache(0, 0);
        const Tensor<T> *q_bias = nullptr;
        const Tensor<T> *k_bias = nullptr;
        const Tensor<T> *v_bias = nullptr;
        const Tensor<T> *o_bias = nullptr;

        try {
            q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
        } catch (const std::out_of_range &) {
        }
        try {
            v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
        } catch (const std::out_of_range &) {
        }

        // 目前只有qkv合并路径适配直接写入
        if (quant_type_ == 0) {
            auto merged_qkv_weight = params_.at("merged_qkv_" + std::to_string(i));
            const Tensor<T> *merged_qkv_bias = nullptr;
            try {
                merged_qkv_bias = &params_.at("merged_qkv_bias_" + std::to_string(i));
            } catch (const std::out_of_range &) {
            }

            // Fused GEMV QKV + RoPE operation
            operators_->gemv_qkv_rope(&hidden_states, &merged_qkv_weight, &q_buf, &k_buf, &v_buf, merged_qkv_bias,
                                      graph.d_rope_offset, &rope_sin_cos_cache_, graph.d_offset_array, i,
                                      n_heads_ * head_dim_, n_kv_heads_ * head_dim_, n_kv_heads_ * head_dim_,
                                      n_heads_, n_kv_heads_, head_dim_, stream, n_layers_, graph.pingpong);
            //                                             head_dim_});

            // Tensor<T> &k = kv_cache->k_cache(i, kv_cache->size());
            // debugPrintTensor(k, "k");
            // debugPrintTensor(k_buf, "k_buf");
            // debugPrintTensor(k_slice, "k_slice");

            // q_buf = q_slice;
            // k_buf = k_slice;
            // v_buf = v_slice;
        } else {
            // AWQ量化路径，需要分别处理QKV投影
            auto q_weight = get_weight(layer_prefix + "self_attn.q_proj");
            auto k_weight = get_weight(layer_prefix + "self_attn.k_proj");
            auto v_weight = get_weight(layer_prefix + "self_attn.v_proj");

            operators_->matmul(&q_buf, &hidden_states, q_weight, q_bias, stream);
            operators_->matmul(&k_buf, &hidden_states, k_weight, k_bias, stream);
            operators_->matmul(&v_buf, &hidden_states, v_weight, v_bias, stream);
        }
        // decode 1
        Tensor<T> q_3d = q_buf.view({1, n_heads_, head_dim_});
        Tensor<T> k_3d = k_buf.view({1, n_kv_heads_, head_dim_});
        Tensor<T> v_3d = v_buf.view({1, n_kv_heads_, head_dim_});

        // 在图中直接写入KV cache，然后通过图节点更新来改变目标地址
        // 获取当前token在KV cache中的位置（图捕获时使用默认位置）

        // size_t current_offset = kv_cache->size() - 1;
        // Tensor<T> &k_slice = kv_cache->k_cache(i, current_offset);
        // Tensor<T> &v_slice = kv_cache->v_cache(i, current_offset);
        // cudaMemcpyAsync(k_slice.data_ptr(), k_3d.data_ptr(), k_3d.numel() * sizeof(T), cudaMemcpyDeviceToDevice,
        //                 stream);
        // cudaMemcpyAsync(v_slice.data_ptr(), v_3d.data_ptr(), v_3d.numel() * sizeof(T), cudaMemcpyDeviceToDevice,
        //                 stream);
        auto [total_K_flat, total_V_flat] = kv_cache->get_contiguous_tensor(i);
        // std::cout << "total_K_flat地址: " << total_K_flat.data_ptr() << std::endl;
        Tensor<T> att_heads(decode_tensor_shape("att_heads"), Device::CUDA, false, graph_layer_tensor_tag("att_heads", i));

        size_t total_seq_len = kv_cache->size();
        // KV cache返回格式：[seq_len, head_dim] 其中 head_dim = n_kv_heads * dqkv
        // Flash Attention期望：[seq_len, n_kv_heads, dqkv]
        Tensor<T> total_K = total_K_flat.view({total_seq_len, n_kv_heads_, head_dim_});
        Tensor<T> total_V = total_V_flat.view({total_seq_len, n_kv_heads_, head_dim_});

        operators_->flash_attention_graph_fixed(q_3d, total_K, total_V, graph.d_output_ptrs, graph.d_segment_info,
                                                n_kv_heads_, stream, graph.pingpong);

        // 使用图优化的gather_fa（固定内存版本）
        operators_->gather_fa_graph_fixed(graph.d_output_ptrs, att_heads, graph.d_segment_info, stream);

        // 输出投影 - 使用tagged memory
        Tensor<T> att_proj(decode_tensor_shape("att_proj"), Device::CUDA, false, graph_layer_tensor_tag("att_proj", i));
        auto o_weight = get_weight(layer_prefix + "self_attn.o_proj");
        Tensor<T> att_heads_reshaped = att_heads.view({seq_len, n_heads_ * head_dim_});
        operators_->matmul(&att_proj, &att_heads_reshaped, o_weight, o_bias, stream);

        // 2.3 后注意力层归一化和残差连接 - 完全仿照普通模式
        auto &ffn_norm_weight = params_.at(layer_prefix + "post_attention_layernorm.weight");
        add_residual_and_norm(&hidden_states, &residual, &att_proj, &ffn_norm_weight, stream);

        // 2.4 MLP前馈网络 - 使用tagged memory
        std::string ffn_tag = graph_layer_tensor_tag("ffn_out", i);
        Tensor<T> gate_buf_silu({seq_len, intermediate_size_}, Device::CUDA, false, graph_layer_tensor_tag("gate_buf_silu", i));

        if (quant_type_ == 0) {
            // Use the fused GEMV MLP kernel for non-quantized models
            auto merged_mlp_weight = params_.at("merged_mlp_" + std::to_string(i));
            operators_->gemv_mlp_fused(&hidden_states, &merged_mlp_weight, &gate_buf_silu, stream);
        } else {
            // Fallback to original implementation for quantized models
            Tensor<T> gate_buf(decode_tensor_shape("gate_buf"), Device::CUDA, false, graph_layer_tensor_tag("gate_buf", i));
            Tensor<T> up_buf(decode_tensor_shape("up_buf"), Device::CUDA, false, graph_layer_tensor_tag("up_buf", i));
            auto gate_weight = get_weight(layer_prefix + "mlp.gate_proj");
            auto up_weight = get_weight(layer_prefix + "mlp.up_proj");
            operators_->matmul(&gate_buf, &hidden_states, gate_weight, nullptr, stream);
            operators_->matmul(&up_buf, &hidden_states, up_weight, nullptr, stream);
            operators_->silu_multiply(&gate_buf_silu, &gate_buf, &up_buf, stream);
        }

        Tensor<T> ffn_out(decode_tensor_shape("ffn_out"), Device::CUDA, false, ffn_tag);
        auto down_weight = get_weight(layer_prefix + "mlp.down_proj");
        operators_->matmul(&ffn_out, &gate_buf_silu, down_weight, nullptr, stream);

        if (i == n_layers_ - 1) {
            // 最后一层的残差连接，直接与最终的RMS norm合并
            auto &norm_weight = params_.at("norm.weight");
            add_residual_and_norm(&hidden_states, &residual, &ffn_out, &norm_weight, stream);
        } else {
            // 非最后一层：残差连接 + 下一层的input_layernorm
            std::string next_layer_prefix = "layers." + std::to_string(i + 1) + ".";
            auto &next_attention_norm_weight = params_.at(next_layer_prefix + "input_layernorm.weight");
            add_residual_and_norm(&hidden_states, &residual, &ffn_out, &next_attention_norm_weight, stream);
        }
    }

    // 最终的LayerNorm已经在最后一层的循环中合并处理了
    Tensor<T> final_h = hidden_states;

    auto lm_head_weight = get_weight("lm_head");
    Tensor<T> logits(decode_tensor_shape("logits"), Device::CUDA, false, graph_tensor_tag("fixed_logits"));
    operators_->matmul(&logits, &final_h, lm_head_weight, nullptr, stream);

    return logits;
}

// 初始化固定内存
template <typename T>
void QwenModel<T>::initialize_graph_fixed_memory() {
    auto &graph = graph_runtime();
    cudaError_t err = cudaMalloc(&graph.d_rope_offset, sizeof(size_t) * 2);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for RoPE offset: " +
                                 std::string(cudaGetErrorString(err)));
    }
    graph.pingpong_index = 0;
    graph.fixed_k_buffers.clear();
    graph.fixed_v_buffers.clear();
    graph.fixed_k_buffers.reserve(n_layers_);
    graph.fixed_v_buffers.reserve(n_layers_);

    for (size_t i = 0; i < n_layers_; i++) {
        Tensor<T> k_buf(decode_tensor_shape("k_buf"), Device::CUDA, false, graph_layer_tensor_tag("fixed_k_buf", i));
        Tensor<T> v_buf(decode_tensor_shape("v_buf"), Device::CUDA, false, graph_layer_tensor_tag("fixed_v_buf", i));

        graph.fixed_k_buffers.push_back(std::move(k_buf));
        graph.fixed_v_buffers.push_back(std::move(v_buf));
    }

    // 分配连续的offset数组，所有层共享
    cudaMalloc(&graph.d_offset_array, n_layers_ * sizeof(int) * 2);
    cudaMalloc(&graph.pingpong, sizeof(int));
    cudaMemset(graph.pingpong, 0, sizeof(int));

    graph.kv_copy_nodes.clear();
    const int max_branches = 3;
    graph.segment_info_tensor = Tensor<int>({2}, Device::CUDA, false, graph_tensor_tag("segment_info"));
    graph.d_segment_info = graph.segment_info_tensor.data_ptr();

    graph.output_ptrs_tensor = Tensor<T *>({max_branches}, Device::CUDA, false, graph_tensor_tag("output_ptrs"));
    graph.d_output_ptrs = graph.output_ptrs_tensor.data_ptr();

    graph.fixed_fa_outputs.clear();
    graph.fixed_fa_outputs.reserve(max_branches);

    for (int i = 0; i < max_branches; i++) {
        Tensor<T> output(decode_tensor_shape("fa_output"), Device::CUDA, false, graph_layer_tensor_tag("fa_output", i));
        graph.fixed_fa_outputs.push_back(std::move(output));
    }

    std::vector<T *> h_output_ptrs(max_branches);
    for (int i = 0; i < max_branches; i++) {
        h_output_ptrs[i] = graph.fixed_fa_outputs[i].data_ptr();
    }

    cudaMemcpyAsync(graph.d_output_ptrs, h_output_ptrs.data(), max_branches * sizeof(T *), cudaMemcpyHostToDevice);

    std::cout << "Graph fixed memory initialized successfully!" << std::endl;
}

template <typename T>
void QwenModel<T>::cleanup_graph_fixed_memory() {
    graph_runtime().release_fixed_memory();
}

template <typename T>
void QwenModel<T>::update_rope_offset(size_t offset, cudaStream_t stream, int pingpong_idx) {
    auto &graph = graph_runtime();
    if (graph.d_rope_offset) {
        cudaError_t err =
            cudaMemcpyAsync(graph.d_rope_offset + pingpong_idx, &offset, sizeof(size_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to update RoPE offset: " + std::string(cudaGetErrorString(err)));
        }
    }
}

template <typename T>
void QwenModel<T>::extract_updateable_nodes() {
    auto &graph = graph_runtime();
    GraphRunner<T>::extract_kv_copy_nodes(graph, n_layers_, n_kv_heads_, head_dim_);
    std::cout << "总共找到 " << graph.kv_copy_nodes.size() << " 个KV复制节点" << std::endl;
}
template <typename T>
void QwenModel<T>::update_graph_kv_addresses(KVCache<T> *kv_cache, size_t offset) {
    GraphRunner<T>::update_kv_copy_nodes(graph_runtime(), kv_cache, offset, n_layers_);
}

template <typename T>
void QwenModel<T>::update_graph_kv_addresses_async_for_next(KVCache<T> *kv_cache, size_t next_offset) {
    auto &graph = graph_runtime();
    if (!graph.graph_initialized || !graph.graph_exec || !kv_cache || graph.kv_copy_nodes.empty()) {
        return;
    }

    // 异步更新图节点参数为下一次执行准备
    // 这个函数在图执行期间被调用，为下一次执行预准备KV地址

    size_t node_idx = 0;
    for (size_t layer = 0; layer < n_layers_ && node_idx < graph.kv_copy_nodes.size(); layer++) {
        // 异步更新K复制节点
        if (node_idx < graph.kv_copy_nodes.size()) {
            cudaMemcpy3DParms k_params;
            cudaError_t get_param_err = cudaGraphMemcpyNodeGetParams(graph.kv_copy_nodes[node_idx], &k_params);
            if (get_param_err == cudaSuccess) {
                // 获取下一次执行的K cache地址
                Tensor<T> &next_k_slice = kv_cache->k_cache(layer, next_offset);
                k_params.dstPtr.ptr = next_k_slice.data_ptr();

                // 异步更新图节点参数
                cudaError_t result =
                    cudaGraphExecMemcpyNodeSetParams(graph.graph_exec, graph.kv_copy_nodes[node_idx], &k_params);

                // 注意：这里不打印错误，因为是异步操作
                // 错误会在下次图执行时体现
            }
            node_idx++;
        }

        // 异步更新V复制节点
        if (node_idx < graph.kv_copy_nodes.size()) {
            cudaMemcpy3DParms v_params;
            cudaError_t get_param_err = cudaGraphMemcpyNodeGetParams(graph.kv_copy_nodes[node_idx], &v_params);
            if (get_param_err == cudaSuccess) {
                // 获取下一次执行的V cache地址
                Tensor<T> &next_v_slice = kv_cache->v_cache(layer, next_offset);
                v_params.dstPtr.ptr = next_v_slice.data_ptr();

                // 异步更新图节点参数
                cudaError_t result =
                    cudaGraphExecMemcpyNodeSetParams(graph.graph_exec, graph.kv_copy_nodes[node_idx], &v_params);
            }
            node_idx++;
        }
    }

    // 注意：这里不需要同步，因为是异步预准备
    // 下次执行时会检查是否已经预准备完成
}

template <typename T>
void QwenModel<T>::update_segment_info(size_t total_seq_len, int layer_idx, cudaStream_t stream, int pp_idx) {
    auto &graph = graph_runtime();
    if (!graph.d_segment_info)
        return;

    int segment_info = total_seq_len;
    cudaMemcpyAsync(graph.d_segment_info + pp_idx, &segment_info, sizeof(int), cudaMemcpyHostToDevice, stream);
}

template <typename T>
void QwenModel<T>::prepare_graph_execution(size_t rope_offset, size_t total_seq_len, cudaStream_t stream, int pp_idx) {
    auto &graph = graph_runtime();
    static thread_local std::vector<int> batch_offsets(n_layers_);
    size_t base_offset = rope_offset * head_dim_ * n_kv_heads_;
    size_t layer_stride = max_position_embeddings_ * head_dim_ * n_kv_heads_;
    for (int i = 0; i < n_layers_; i++) {
        batch_offsets[i] = base_offset + i * layer_stride;
    }
    cudaMemcpyAsync(graph.d_offset_array + pp_idx * n_layers_, batch_offsets.data(), n_layers_ * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    update_rope_offset(rope_offset, stream, pp_idx);
    cudaMemcpyAsync(graph.d_segment_info + pp_idx, &total_seq_len, sizeof(int), cudaMemcpyHostToDevice, stream);
}

template <typename T>
void QwenModel<T>::initialize_cuda_graph_with_kv_cache(KVCache<T> *kv_cache) {
    auto &graph = graph_runtime();
    if (graph.graph_initialized) {
        return;  // 已经初始化过了
    }

    std::cout << "使用真实KV cache初始化CUDA图..." << std::endl;

    // 初始化图执行所需的固定内存
    initialize_graph_fixed_memory();

    // 创建固定的输入输出张量 - 关键修复：使用tagged memory确保地址一致性
    graph.graph_input_tensor =
        Tensor<uint32_t>({1}, Device::CUDA, false, graph_tensor_tag("input_token"));  // 单token输入，使用独立的标签
    graph.graph_output_tensor =
        Tensor<T>(decode_tensor_shape("logits"), Device::CUDA, false, graph_tensor_tag("fixed_logits"));

    uint32_t init_token = 9707;  // 使用一个有效的token ID
    cudaMemcpy(graph.graph_input_tensor.data_ptr(), &init_token, sizeof(uint32_t), cudaMemcpyHostToDevice);
    std::cout << "初始化graph_input_tensor_为token: " << init_token << std::endl;

    GraphRunner<T>::initialize(
        graph, "Qwen",
        [&]() {
            std::cout << "使用真实KV cache进行预热运行..." << std::endl;
            size_t default_offset = kv_cache->size() - 1;
            size_t default_total_seq_len = kv_cache->size();
            update_rope_offset(default_offset, nullptr, graph.pingpong_index);
            update_segment_info(default_total_seq_len, 0, nullptr, graph.pingpong_index);
            cudaDeviceSynchronize();
            Tensor<T> warmup_output = forward_for_graph(&graph.graph_input_tensor, kv_cache, nullptr);
            (void)warmup_output;
            cudaDeviceSynchronize();
            std::cout << "预热运行完成！" << std::endl;
        },
        [&]() {
            std::cout << "开始CUDA图捕获..." << std::endl;
            std::cout << "图捕获前graph_input_tensor_地址: " << graph.graph_input_tensor.data_ptr() << std::endl;
            uint32_t capture_token = 4;
            cudaMemcpy(graph.graph_input_tensor.data_ptr(), &capture_token, sizeof(uint32_t), cudaMemcpyHostToDevice);
            graph.graph_output_tensor =
                Tensor<T>(decode_tensor_shape("logits"), Device::CUDA, false, graph_tensor_tag("fixed_logits"));
            return forward_for_graph(&graph.graph_input_tensor, kv_cache, graph.graph_stream);
        });

    extract_updateable_nodes();
    std::cout << "图捕获后graph_input_tensor_地址: " << graph.graph_input_tensor.data_ptr() << std::endl;
    std::cout << "CUDA图初始化成功！找到 " << graph.kv_copy_nodes.size() << " 个可更新节点" << std::endl;
}

// -------------------------------
// 逐层输出保存功能：保存中间张量到二进制文件用于分析
// -------------------------------
template <typename T>
void QwenModel<T>::save_tensor_to_binary(const Tensor<T> &tensor, const std::string &filename) {
    // 创建目录（如果不存在）
    //     std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    //     if (!dir_path.empty()) {
    //         std::string mkdir_cmd = "mkdir -p " + dir_path;
    //         system(mkdir_cmd.c_str());
    //     }

    //     // 将GPU张量数据复制到CPU
    //     std::vector<T> host_data(tensor.numel());
    //     cudaMemcpy(host_data.data(), tensor.data_ptr(), tensor.numel() *
    //     sizeof(T), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();

    //     // 保存到二进制文件
    //     std::ofstream file(filename, std::ios::binary);
    //     if (!file.is_open()) {
    //         throw std::runtime_error("Failed to open file for writing: " +
    //         filename);
    //     }

    //     // 写入张量形状信息
    //     size_t ndim = tensor.sizes().size();
    //     file.write(reinterpret_cast<const char*>(&ndim), sizeof(size_t));
    //     for (size_t dim : tensor.sizes()) {
    //         file.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
    //     }

    //     // 写入数据类型大小
    //     size_t dtype_size = sizeof(T);
    //     file.write(reinterpret_cast<const char*>(&dtype_size), sizeof(size_t));

    //     // 写入张量数据
    //     file.write(reinterpret_cast<const char*>(host_data.data()),
    //     tensor.numel() * sizeof(T)); file.close();

    //     std::cout << "已保存张量到: " << filename << " (形状: [";
    //     for (size_t i = 0; i < tensor.sizes().size(); i++) {
    //         std::cout << tensor.sizes()[i];
    //         if (i < tensor.sizes().size() - 1) std::cout << ", ";
    //     }
    //     std::cout << "], 元素数: " << tensor.numel() << ")" << std::endl;
}

// 保存uint32_t张量的专用函数
template <typename T>
void QwenModel<T>::save_uint32_tensor_to_binary(const Tensor<uint32_t> &tensor, const std::string &filename) {
    //   // 创建目录（如果不存在）
    //   std::string dir_path = filename.substr(0, filename.find_last_of('/'));
    //   if (!dir_path.empty()) {
    //     std::string mkdir_cmd = "mkdir -p " + dir_path;
    //     system(mkdir_cmd.c_str());
    //   }

    //   // 将GPU张量数据复制到CPU
    //   std::vector<uint32_t> host_data(tensor.numel());
    //   cudaMemcpy(host_data.data(), tensor.data_ptr(),
    //              tensor.numel() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //   cudaDeviceSynchronize();

    //   // 保存到二进制文件
    //   std::ofstream file(filename, std::ios::binary);
    //   if (!file.is_open()) {
    //     throw std::runtime_error("Failed to open file for writing: " +
    //     filename);
    //   }

    //   // 写入张量形状信息
    //   size_t ndim = tensor.sizes().size();
    //   file.write(reinterpret_cast<const char *>(&ndim), sizeof(size_t));
    //   for (size_t dim : tensor.sizes()) {
    //     file.write(reinterpret_cast<const char *>(&dim), sizeof(size_t));
    //   }

    //   // 写入数据类型大小
    //   size_t dtype_size = sizeof(uint32_t);
    //   file.write(reinterpret_cast<const char *>(&dtype_size), sizeof(size_t));

    //   // 写入张量数据
    //   file.write(reinterpret_cast<const char *>(host_data.data()),
    //              tensor.numel() * sizeof(uint32_t));
    //   file.close();

    //   std::cout << "已保存uint32张量到: " << filename << " (形状: [";
    //   for (size_t i = 0; i < tensor.sizes().size(); i++) {
    //     std::cout << tensor.sizes()[i];
    //     if (i < tensor.sizes().size() - 1) std::cout << ", ";
    //   }
    //   std::cout << "], 元素数: " << tensor.numel() << ")" << std::endl;
}

// -------------------------------
// CPU Sample 相关新方法实现
// -------------------------------

// 只执行forward计算，返回logits，不进行sample
template <typename T>
Tensor<T> QwenModel<T>::forward_logits_only(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    if (device_ == Device::CUDA) {
        return forward_cuda(input, kv_cache);
    }
    return forward_generic(input, kv_cache);
}

// CUDA图模式下只执行forward计算，返回logits
template <typename T>
Tensor<T> QwenModel<T>::forward_for_graph_logits_only(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    auto &graph = graph_runtime();
    // 延迟初始化CUDA图，使用真实的KV cache
    if (!graph.graph_initialized) {
        initialize_cuda_graph_with_kv_cache(kv_cache);
    }

    size_t rope_offset = kv_cache->size() - 1;  // 当前token的位置索引
    size_t total_seq_len = kv_cache->size();    // flash attention应该看到包含当前token的完整数据
    if (kv_cache->size() > graph.last_kv_cache_size + 1) {
        std::cout << "检测到新轮对话开始，KV cache从 " << graph.last_kv_cache_size - 1 << " 跳跃到 "
                  << kv_cache->size()
                  << std::endl;
        prepare_graph_execution(rope_offset, total_seq_len, graph.graph_stream, graph.pingpong_index);
    } else {
        cudaStreamSynchronize(graph.prep_stream);
    }
    graph.last_kv_cache_size = kv_cache->size();
    
    // cudaStreamSynchronize(graph_stream_);
    GraphRunner<T>::launch(graph, "Qwen");
    // compute_next_offsets_async(total_seq_len);

    prepare_graph_execution(rope_offset + 1, total_seq_len + 1, graph.prep_stream, graph.pingpong_index);
    cudaMemcpyAsync(graph.pingpong, &graph.pingpong_index, sizeof(int), cudaMemcpyHostToDevice, graph.graph_stream);
    return graph.graph_output_tensor;
}

// 在CPU上执行sample操作
template <typename T>
uint32_t QwenModel<T>::sample_cpu(const Tensor<T> &gpu_logits, float temperature, float top_p, size_t top_k) {
    // 1. 将logits拷贝到CPU（需要创建可修改的副本）
    Tensor<T> cpu_logits = gpu_logits;
    cpu_logits.cpu();  // 移动到CPU

    // 2. 确保CPU算子已初始化
    if (!cpu_operators_) {
        cpu_operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CPU);
    }

    // 3. 调用CPU版本的sample
    return cpu_operators_->sample_cpu(std::move(cpu_logits), temperature, top_p, top_k);
}

// 分配GPU内存并返回结果指针
template <typename T>
uint32_t *QwenModel<T>::allocate_gpu_result(uint32_t result) {
    auto &graph = graph_runtime();
    if (graph.graph_input_tensor.numel() == 0 || graph.graph_input_tensor.device() != Device::CUDA) {
        graph.graph_input_tensor = Tensor<uint32_t>({1}, Device::CUDA, false, "qwen_sample_result");
    }
    cudaMemcpy(graph.graph_input_tensor.data_ptr(), &result, sizeof(uint32_t), cudaMemcpyHostToDevice);
    return graph.graph_input_tensor.data_ptr();
}

// -------------------------------
// 计算下一轮的offset（异步）
// -------------------------------
template <typename T>
void QwenModel<T>::compute_next_offsets_async(int offset) {
    // 预计算下一轮执行所需的offset值

    size_t next_rope_offset = offset;  // 下一个token的位置
    size_t base_offset = next_rope_offset * head_dim_ * n_kv_heads_;
    size_t layer_stride = max_position_embeddings_ * head_dim_ * n_kv_heads_;

    // 确保next_batch_offsets_有正确的大小
    next_batch_offsets_.resize(n_layers_);

    // 向量化计算所有layer的offset
    for (int i = 0; i < n_layers_; i++) {
        next_batch_offsets_[i] = base_offset + i * layer_stride;
    }

    offsets_prepared_ = true;
}

// -------------------------------
// 应用预准备的offset
// -------------------------------
template <typename T>
void QwenModel<T>::apply_prepared_offsets() {
    auto &graph = graph_runtime();
    if (!offsets_prepared_ || next_batch_offsets_.empty()) {
        return;
    }

    // 将预计算的offset复制到设备端数组
    if (graph.d_offset_array) {
        cudaMemcpyAsync(graph.d_offset_array, next_batch_offsets_.data(), n_layers_ * sizeof(int),
                        cudaMemcpyHostToDevice, graph.graph_stream);
    }

    offsets_prepared_ = false;
}

// -------------------------------
// Sampling methods
// -------------------------------
template <typename T>
void QwenModel<T>::set_sample_mode(SampleMode mode) {
    sample_mode_ = mode;
}

template <typename T>
uint32_t *QwenModel<T>::sample_unified(const Tensor<T> &logits, float temperature, float top_p, size_t top_k,
                                       KVCache<T> *kv_cache, curandState *d_states, cudaStream_t stream) {
    switch (sample_mode_) {
        case SampleMode::CPU:
            return sample_with_cpu_only(logits, temperature, top_p, top_k);
        case SampleMode::GPU:
        case SampleMode::GPU_WITH_ASYNC_PREPARE:
        default:
            return operators_->sample(Tensor<T>(logits), temperature, top_p, top_k, d_states, stream);
    }
}

template <typename T>
uint32_t *QwenModel<T>::sample_with_cpu_only(const Tensor<T> &logits, float temperature, float top_p, size_t top_k) {
    uint32_t cpu_result = sample_cpu(logits, temperature, top_p, top_k);
    if (device_ == Device::CPU) {
        return new uint32_t(cpu_result);
    }
    return allocate_gpu_result(cpu_result);
}

template <typename T>
uint32_t *QwenModel<T>::sample_with_metadata_update(const Tensor<T> &logits, float temperature, float top_p,
                                                    size_t top_k, KVCache<T> *kv_cache) {
    // This method can be used for any sampling that needs KV cache metadata updates
    return sample_unified(logits, temperature, top_p, top_k, kv_cache);
}

// -------------------------------
// 显式模板实例化
// -------------------------------
template class QwenModel<float>;
template class QwenModel<__nv_bfloat16>;
