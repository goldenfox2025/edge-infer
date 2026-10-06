#pragma once

#include <limits>

#include "execution/cuda_workspace_arena.hpp"
#include "execution/workspace_plan.hpp"
#include "inference.hpp"
#include "models/qwen3_model.hpp"
#include "operators/cuda/execution.hpp"

enum class DecoderMode { Decode, Prefill, Graph };

// Logical RoPE positions and physical KV slots are deliberately independent.
struct DecoderStep {
  size_t cache_offset = 0;
  size_t position_offset = 0;
  DecoderMode mode = DecoderMode::Decode;
  const size_t* device_position = nullptr;
  const size_t* device_cache_offset = nullptr;
  int* device_lengths = nullptr;
  int* pingpong = nullptr;
};

template <typename T>
struct DecoderBuffers {
  TensorView<T, 2> residual, attention_norm, ffn_norm, final_norm;
  TensorView<T, 2> q, k, v, attention, projected, gate, up, ffn, logits;
  TensorView<float, 1> attention_scratch;
};

// Inclusive lifetimes describe one block. Blocks execute sequentially and
// reuse this layout; residual remains live across every block and the head.
template <typename T>
WorkspacePlan plan_decoder_workspace(const Qwen3Config& c, size_t rows) {
  if (!rows || rows > c.max_position_embeddings)
    throw std::invalid_argument("Decoder workspace rows exceed model context");
  WorkspacePlanner p;
  auto value = [&](const char* name, size_t width, size_t first, size_t last) {
    if (width > std::numeric_limits<size_t>::max() / sizeof(T) / rows ||
        width > static_cast<size_t>(std::numeric_limits<int>::max()) / rows)
      throw std::overflow_error("Decoder workspace extent overflow");
    p.add_request(name, rows * width * sizeof(T), first, last);
  };
  value("residual", c.hidden_size, 0, 17);
  value("attention_norm", c.hidden_size, 1, 4);
  value("q", c.n_heads * c.head_dim, 2, 7);
  value("k", c.n_kv_heads * c.head_dim, 3, 6);
  value("v", c.n_kv_heads * c.head_dim, 4, 6);
  value("attention", c.n_heads * c.head_dim, 7, 8);
  value("projected", c.hidden_size, 8, 9);
  value("ffn_norm", c.hidden_size, 10, 12);
  value("gate", c.intermediate_size, 11, 15);
  value("up", c.intermediate_size, 12, 14);
  value("ffn", c.hidden_size, 15, 16);
  value("final_norm", c.hidden_size, 17, 18);
  value("logits", c.vocab_size, 18, 19);
  // Fast decode splits attention into five branches. Prefill needs no scratch.
  const size_t scratch = rows == 1 ? 5 * c.n_heads * (c.head_dim + 2) : 1;
  if (scratch > std::numeric_limits<size_t>::max() / sizeof(float))
    throw std::overflow_error("Decoder attention scratch extent overflow");
  p.add_request("attention_scratch", scratch * sizeof(float), 7, 7);
  return p.build();
}

template <typename T>
DecoderBuffers<T> resolve_decoder_buffers(const Qwen3Config& c, size_t rows,
                                          const WorkspacePlan& plan, CudaWorkspaceArena& arena) {
  arena.reserve_for_plan(plan);
  auto take = [&](const char* name, size_t width) -> TensorView<T, 2> {
    return {arena.template ptr_at<T>(plan.at(name).offset), {rows, width}, {width, 1}};
  };
  return {take("residual", c.hidden_size),
          take("attention_norm", c.hidden_size),
          take("ffn_norm", c.hidden_size),
          take("final_norm", c.hidden_size),
          take("q", c.n_heads * c.head_dim),
          take("k", c.n_kv_heads * c.head_dim),
          take("v", c.n_kv_heads * c.head_dim),
          take("attention", c.n_heads * c.head_dim),
          take("projected", c.hidden_size),
          take("gate", c.intermediate_size),
          take("up", c.intermediate_size),
          take("ffn", c.hidden_size),
          take("logits", c.vocab_size),
          {arena.template ptr_at<float>(plan.at("attention_scratch").offset),
           {plan.at("attention_scratch").bytes / sizeof(float)},
           {1}}};
}

template <typename T, size_t Rank>
inline op::ArrayView<T> flat(TensorView<T, Rank> v) noexcept {
  return {v.data, v.numel()};
}

template <typename T>
inline void execute_linear(const op::cuda::ExecutionContext& ctx, TensorView<const T, 2> input,
                           const PreparedLinear<T>& weight, TensorView<T, 2> output) {
  if (weight.quantized)
    op::cuda::awq_linear<T>(ctx, input, weight.awq, output);
  else
    op::cuda::linear<T>(ctx, input, weight.dense, output);
}

// Embeddings are already in b.residual. This transformer backbone has no token
// lookup, LM head or sampling policy. Speech adapters can supply embeddings and
// independent positions, then consume the returned normalized hidden states.
template <typename T>
TensorView<T, 2> execute_decoder(const op::cuda::ExecutionContext& ctx, const Qwen3Model<T>& model,
                                 KVCache<T>& cache, DecoderBuffers<T>& b, const DecoderStep& step,
                                 float** graph_branches = nullptr) {
  const auto& c = model.config();
  const size_t rows = b.residual.shape[0];
  auto norm = [&](TensorView<T, 2> out, TensorView<const T, 2> in, TensorView<const T, 1> weight,
                  size_t n, size_t width) {
    op::cuda::rms_norm<T>(ctx, flat(in), flat(weight), flat(out), n, width, c.rms_norm_eps);
  };
  for (size_t i = 0; i < model.layers().size(); ++i) {
    const auto& w = model.layers()[i];
    norm(b.attention_norm, b.residual.as_const(), w.attention_norm, rows, c.hidden_size);
    execute_linear(ctx, b.attention_norm.as_const(), w.q, b.q);
    execute_linear(ctx, b.attention_norm.as_const(), w.k, b.k);
    execute_linear(ctx, b.attention_norm.as_const(), w.v, b.v);
    if (c.qk_norm) {
      norm(b.q, b.q.as_const(), w.q_norm, rows * c.n_heads, c.head_dim);
      norm(b.k, b.k.as_const(), w.k_norm, rows * c.n_kv_heads, c.head_dim);
    }
    auto q = b.q.template reshape_contiguous<3>({rows, c.n_heads, c.head_dim});
    auto k = b.k.template reshape_contiguous<3>({rows, c.n_kv_heads, c.head_dim});
    auto v = b.v.template reshape_contiguous<3>({rows, c.n_kv_heads, c.head_dim});
    // Every execution mode uses the same prepared trigonometric values.
    op::cuda::rope_precomputed<T>(ctx, q, model.rope_view(), step.position_offset,
                                 step.device_position, step.pingpong);
    op::cuda::rope_precomputed<T>(ctx, k, model.rope_view(), step.position_offset,
                                 step.device_position, step.pingpong);
    op::cuda::store_kv<T>(ctx, b.k.as_const(), cache.k_capacity_view(i), step.cache_offset,
                          step.device_cache_offset);
    op::cuda::store_kv<T>(ctx, b.v.as_const(), cache.v_capacity_view(i), step.cache_offset,
                          step.device_cache_offset);
    const bool graph = step.mode == DecoderMode::Graph;
    auto ck = (graph ? cache.k_capacity_view(i) : cache.k_view(i))
                  .template reshape_contiguous<3>(
                      {graph ? cache.get_max_seq_len() : cache.size(), c.n_kv_heads, c.head_dim});
    auto cv = (graph ? cache.v_capacity_view(i) : cache.v_view(i))
                  .template reshape_contiguous<3>(
                      {graph ? cache.get_max_seq_len() : cache.size(), c.n_kv_heads, c.head_dim});
    auto att = b.attention.template reshape_contiguous<3>({rows, c.n_heads, c.head_dim});
    if (graph)
      op::cuda::attention_graph<T>(ctx, q.as_const(), ck.as_const(), cv.as_const(), att,
                                   graph_branches, step.device_lengths, step.pingpong);
    else if (step.mode == DecoderMode::Prefill)
      op::cuda::attention_prefill<T>(ctx, q.as_const(), ck.as_const(), cv.as_const(), att,
                                     step.cache_offset);
    else
      op::cuda::attention_decode<T>(ctx, q.as_const(), ck.as_const(), cv.as_const(), att,
                                    b.attention_scratch);
    execute_linear(ctx, b.attention.as_const(), w.o, b.projected);
    op::cuda::add<T>(ctx, flat(b.residual.as_const()), flat(b.projected.as_const()),
                     flat(b.residual));
    norm(b.ffn_norm, b.residual.as_const(), w.ffn_norm, rows, c.hidden_size);
    execute_linear(ctx, b.ffn_norm.as_const(), w.gate, b.gate);
    execute_linear(ctx, b.ffn_norm.as_const(), w.up, b.up);
    op::cuda::silu_multiply<T>(flat(b.gate.as_const()), flat(b.up.as_const()), flat(b.gate),
                               ctx.stream);
    execute_linear(ctx, b.gate.as_const(), w.down, b.ffn);
    op::cuda::add<T>(ctx, flat(b.residual.as_const()), flat(b.ffn.as_const()), flat(b.residual));
  }
  norm(b.final_norm, b.residual.as_const(), model.output_norm_view(), rows, c.hidden_size);
  return b.final_norm;
}
