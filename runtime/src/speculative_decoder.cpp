#include "speculative_decoder.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "operators/cuda/random.hpp"

namespace {

std::shared_ptr<BaseModel> require_speculative_model(std::shared_ptr<BaseModel> model) {
  if (!model) throw std::invalid_argument("Speculative decoder requires two models");
  auto executor = model->fork_executor();
  return executor ? std::move(executor) : std::move(model);
}

size_t checked_spec_length(size_t count) {
  if (!count || count > 8) throw std::invalid_argument("Speculative length must be between 1 and 8");
  return count;
}

class SpeculativeDeviceScope {
 public:
  explicit SpeculativeDeviceScope(int device) : device_(device) {
    checkCudaErrors(cudaGetDevice(&previous_));
    if (previous_ != device_) checkCudaErrors(cudaSetDevice(device_));
  }
  ~SpeculativeDeviceScope() {
    if (previous_ != device_) cudaSetDevice(previous_);
  }
 private:
  int device_;
  int previous_ = -1;
};

void validate_speculative_token(uint32_t token, size_t vocabulary) {
  if (token >= vocabulary) throw std::runtime_error("Speculative token exceeds the shared vocabulary");
}

}  // namespace

template <typename T>
SpeculativeDecoder<T>::SpeculativeDecoder(std::shared_ptr<BaseModel> target_model,
                                          std::shared_ptr<BaseModel> draft_model,
                                          size_t spec_length, size_t thread_count)
    : target_model_(require_speculative_model(std::move(target_model))),
      draft_model_(require_speculative_model(std::move(draft_model))),
      target_kv_cache_(target_model_->get_n_layers(), target_model_->get_max_seq_len(),
                       target_model_->get_head_dim() * target_model_->get_n_kv_heads(), Device::CUDA),
      draft_kv_cache_(draft_model_->get_n_layers(), draft_model_->get_max_seq_len(),
                      draft_model_->get_head_dim() * draft_model_->get_n_kv_heads(), Device::CUDA),
      spec_length_(checked_spec_length(spec_length)), adaptive_spec_length_(spec_length_) {
  // Retain the frontend constructor parameter for source compatibility. Native
  // speculative execution submits directly to prepared CUDA model sessions.
  (void)thread_count;
  if (target_model_.get() == draft_model_.get())
    throw std::invalid_argument("Target and draft require independent model sessions");
  if (target_model_->get_vocab_size() != draft_model_->get_vocab_size())
    throw std::invalid_argument("Target and draft must share a token vocabulary");
  target_spec_model_ = std::dynamic_pointer_cast<SpeculativeModel<T>>(target_model_);
  draft_spec_model_ = std::dynamic_pointer_cast<SpeculativeModel<T>>(draft_model_);
  if (!target_spec_model_ || !draft_spec_model_)
    throw std::invalid_argument("Speculative models must expose borrowed logits");
  if (target_model_->device() != Device::CUDA) target_model_->cuda();
  if (draft_model_->device() != Device::CUDA) draft_model_->cuda();
  try {
    init_cuda_resources();
  } catch (...) {
    free_cuda_resources();
    throw;
  }
}

template <typename T>
SpeculativeDecoder<T>::~SpeculativeDecoder() {
  free_cuda_resources();
  // Release borrowed-cache sessions before the target/draft KV members.
  target_spec_model_.reset();
  draft_spec_model_.reset();
  target_model_.reset();
  draft_model_.reset();
}

template <typename T>
void SpeculativeDecoder<T>::init_cuda_resources() {
  checkCudaErrors(cudaGetDevice(&cuda_device_id_));
  checkCudaErrors(cudaStreamCreateWithFlags(&draft_stream_, cudaStreamNonBlocking));
  checkCudaErrors(cudaStreamCreateWithFlags(&verify_stream_, cudaStreamNonBlocking));
  draft_context_ = op::cuda::prepare_execution_context(nullptr, draft_stream_);
  verify_context_ = draft_context_;
  verify_context_.stream = verify_stream_;
  draft_sampling_plan_ = op::cuda::prepare_sampling(draft_context_, draft_model_->get_vocab_size());
  target_sampling_plan_ = op::cuda::prepare_sampling(verify_context_, target_model_->get_vocab_size());
  draft_sampling_storage_.reserve(draft_sampling_plan_.total_bytes);
  target_sampling_storage_.reserve(target_sampling_plan_.total_bytes);
  draft_scratch_ = {draft_sampling_storage_.template ptr_at<unsigned char>(0),
                    {draft_sampling_plan_.total_bytes}, {1}};
  target_scratch_ = {target_sampling_storage_.template ptr_at<unsigned char>(0),
                     {target_sampling_plan_.total_bytes}, {1}};

  WorkspacePlanner planner;
  planner.add_request("draft_rng", sizeof(curandState), 0, 1);
  planner.add_request("target_rng", sizeof(curandState), 0, 1);
  planner.add_request("draft_tokens", (MAX_SPEC_LENGTH + 1) * sizeof(uint32_t), 0, 1);
  planner.add_request("draft_probabilities", (MAX_SPEC_LENGTH + 1) * sizeof(float), 0, 1);
  planner.add_request("random_values", MAX_SPEC_LENGTH * sizeof(float), 0, 1);
  planner.add_request("target_tokens", MAX_SPEC_LENGTH * sizeof(uint32_t), 0, 1);
  planner.add_request("target_probabilities", MAX_SPEC_LENGTH * sizeof(float), 0, 1);
  const auto plan = planner.build();
  resources_.reserve_for_plan(plan);
  draft_states_ = resources_.template ptr_at<curandState>(plan.at("draft_rng").offset);
  target_states_ = resources_.template ptr_at<curandState>(plan.at("target_rng").offset);
  draft_tokens_ = resources_.template ptr_at<uint32_t>(plan.at("draft_tokens").offset);
  draft_probabilities_ = resources_.template ptr_at<float>(plan.at("draft_probabilities").offset);
  random_values_ = resources_.template ptr_at<float>(plan.at("random_values").offset);
  target_tokens_ = resources_.template ptr_at<uint32_t>(plan.at("target_tokens").offset);
  target_probabilities_ = resources_.template ptr_at<float>(plan.at("target_probabilities").offset);
  const size_t prompt_capacity = std::min(target_kv_cache_.get_max_seq_len(), draft_kv_cache_.get_max_seq_len());
  prompt_storage_.reserve(prompt_capacity * sizeof(uint32_t));
  token_inputs_.reserve(MAX_SPEC_LENGTH);
  batch_inputs_.reserve(MAX_SPEC_LENGTH);
  for (size_t i = 0; i < MAX_SPEC_LENGTH; ++i) {
    token_inputs_.push_back(Tensor<uint32_t>::from_external_buffer(draft_tokens_ + i, {1}, Device::CUDA));
    batch_inputs_.push_back(Tensor<uint32_t>::from_external_buffer(draft_tokens_, {i + 1}, Device::CUDA));
  }
  const auto seed = static_cast<unsigned long long>(
      std::chrono::system_clock::now().time_since_epoch().count());
  op::cuda::init_curand(draft_states_, seed, 0, draft_stream_);
  op::cuda::init_curand(target_states_, seed, 1, verify_stream_);
  checkCudaErrors(cudaStreamSynchronize(draft_stream_));
  checkCudaErrors(cudaStreamSynchronize(verify_stream_));
}

template <typename T>
void SpeculativeDecoder<T>::free_cuda_resources() noexcept {
  if (cuda_device_id_ < 0) return;
  int previous = cuda_device_id_;
  cudaGetDevice(&previous);
  if (previous != cuda_device_id_) cudaSetDevice(cuda_device_id_);
  // Model logits are synchronized before returning through SpeculativeModel.
  // Finish every owned sampling/copy stream before releasing its private buffers.
  if (draft_stream_) cudaStreamSynchronize(draft_stream_);
  if (verify_stream_) cudaStreamSynchronize(verify_stream_);
  token_inputs_.clear();
  batch_inputs_.clear();
  resources_.release();
  prompt_storage_.release();
  draft_sampling_storage_.release();
  target_sampling_storage_.release();
  if (draft_stream_) cudaStreamDestroy(draft_stream_);
  if (verify_stream_) cudaStreamDestroy(verify_stream_);
  draft_stream_ = verify_stream_ = nullptr;
  if (previous != cuda_device_id_) cudaSetDevice(previous);
  cuda_device_id_ = -1;
}

template <typename T>
void SpeculativeDecoder<T>::generate_draft_tokens(size_t count, bool probabilities,
                                                  float temperature, float top_p, size_t top_k) {
  if (!count || count > MAX_SPEC_LENGTH) throw std::invalid_argument("Invalid draft capacity");
  for (size_t i = 0; i < count; ++i) {
    draft_kv_cache_.resize(draft_kv_cache_.size() + 1);
    auto logits = draft_spec_model_->speculative_forward_logits(&token_inputs_[i], &draft_kv_cache_);
    if (logits.shape[0] != 1) throw std::runtime_error("Draft decode must return one logit row");
    TensorView<float, 1> probability_output;
    if (probabilities) probability_output = {draft_probabilities_ + i + 1, {1}, {1}};
    op::cuda::sample<T>(draft_context_, logits.as_const(), {draft_tokens_ + i + 1, {1}, {1}},
                       probability_output, draft_scratch_, draft_sampling_plan_, temperature,
                       top_p, probabilities ? top_k : 1, draft_states_);
    // The next model call runs on its own stream; it must observe this sampled input.
    checkCudaErrors(cudaStreamSynchronize(draft_stream_));
  }
}

template <typename T>
TensorView<T, 2> SpeculativeDecoder<T>::verify_logits(size_t count) {
  target_kv_cache_.resize(target_kv_cache_.size() + count);
  // N proposals are verified from N inputs [pending, y1, ..., y(N-1)].
  // The trailing proposal yN is retained for comparison, not fed as an input.
  auto logits = target_spec_model_->speculative_prefill_logits(&batch_inputs_[count - 1], &target_kv_cache_);
  if (logits.shape[0] != count) throw std::runtime_error("Target verification returned an invalid batch extent");
  return logits;
}

template <typename T>
size_t SpeculativeDecoder<T>::verify_greedy(size_t count, float temperature, float top_p,
                                            std::vector<uint32_t>& verified) {
  const size_t original = target_kv_cache_.size();
  auto logits = verify_logits(count);
  op::cuda::sample<T>(verify_context_, logits.as_const(), {target_tokens_, {count}, {1}}, {},
                     target_scratch_, target_sampling_plan_, temperature, top_p, 1, target_states_);
  std::array<uint32_t, MAX_SPEC_LENGTH> target{};
  std::array<uint32_t, MAX_SPEC_LENGTH> draft{};
  checkCudaErrors(cudaMemcpyAsync(target.data(), target_tokens_, count * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, verify_stream_));
  checkCudaErrors(cudaMemcpyAsync(draft.data(), draft_tokens_ + 1, count * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, verify_stream_));
  checkCudaErrors(cudaStreamSynchronize(verify_stream_));
  verified.clear();
  size_t accepted = 0;
  for (size_t i = 0; i < count; ++i) {
    validate_speculative_token(target[i], target_model_->get_vocab_size());
    verified.push_back(target[i]);
    if (target[i] != draft[i]) break;
    ++accepted;
  }
  // Retain the pending input plus every accepted prefix input. A target
  // replacement remains pending for the next iteration in both sessions.
  const size_t retained = original + verified.size();
  target_kv_cache_.resize(retained);
  draft_kv_cache_.resize(retained);
  return accepted;
}

template <typename T>
size_t SpeculativeDecoder<T>::verify_probability_ratio(size_t count, float temperature,
                                                       float top_p, size_t top_k,
                                                       std::vector<uint32_t>& verified) {
  const size_t original = target_kv_cache_.size();
  auto logits = verify_logits(count);
  op::cuda::generate_random_values(random_values_, count, target_states_, verify_stream_);
  std::array<uint32_t, MAX_SPEC_LENGTH> draft{};
  std::array<float, MAX_SPEC_LENGTH> random{};
  std::array<float, MAX_SPEC_LENGTH> draft_probabilities{};
  std::array<float, MAX_SPEC_LENGTH> target_probabilities{};
  checkCudaErrors(cudaMemcpyAsync(draft.data(), draft_tokens_ + 1, count * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, verify_stream_));
  checkCudaErrors(cudaMemcpyAsync(random.data(), random_values_, count * sizeof(float),
                                  cudaMemcpyDeviceToHost, verify_stream_));
  checkCudaErrors(cudaMemcpyAsync(draft_probabilities.data(), draft_probabilities_ + 1, count * sizeof(float),
                                  cudaMemcpyDeviceToHost, verify_stream_));
  checkCudaErrors(cudaStreamSynchronize(verify_stream_));
  for (size_t i = 0; i < count; ++i) {
    validate_speculative_token(draft[i], target_model_->get_vocab_size());
    op::cuda::token_probability<T>(verify_context_, logits.as_const(), i, draft[i], target_probabilities_ + i);
  }
  checkCudaErrors(cudaMemcpyAsync(target_probabilities.data(), target_probabilities_, count * sizeof(float),
                                  cudaMemcpyDeviceToHost, verify_stream_));
  checkCudaErrors(cudaStreamSynchronize(verify_stream_));
  verified.clear();
  size_t accepted = 0;
  for (size_t i = 0; i < count; ++i) {
    const float p = target_probabilities[i], q = draft_probabilities[i];
    if (!std::isfinite(p) || !std::isfinite(q) || p < 0.0f || p > 1.0f || q <= 0.0f || q > 1.0f)
      throw std::runtime_error("Speculative acceptance requires finite, valid probabilities");
    // Experimental policy: p is the unscaled target softmax probability, q
    // is the sampled draft distribution, and rejection uses target sampling.
    // This is not exact residual-distribution speculative sampling.
    if (random[i] <= std::min(1.0f, p / q)) {
      verified.push_back(draft[i]);
      ++accepted;
    } else {
      op::cuda::sample<T>(verify_context_, logits.subview({i, 0}, {1, logits.shape[1]}).as_const(),
                         {target_tokens_, {1}, {1}}, {}, target_scratch_, target_sampling_plan_,
                         temperature, top_p, top_k, target_states_);
      uint32_t replacement = 0;
      checkCudaErrors(cudaMemcpyAsync(&replacement, target_tokens_, sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, verify_stream_));
      checkCudaErrors(cudaStreamSynchronize(verify_stream_));
      validate_speculative_token(replacement, target_model_->get_vocab_size());
      verified.push_back(replacement);
      break;
    }
  }
  const size_t retained = original + verified.size();
  target_kv_cache_.resize(retained);
  draft_kv_cache_.resize(retained);
  return accepted;
}

template <typename T>
void SpeculativeDecoder<T>::generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                                                    float temperature, float top_p, size_t top_k,
                                                    std::function<void(uint32_t)> callback) {
  if (input_ids.empty()) throw std::invalid_argument("Prompt must be nonempty");
  if (!callback) throw std::invalid_argument("Speculative generation requires a callback");
  const size_t capacity = std::min(target_kv_cache_.get_max_seq_len(), draft_kv_cache_.get_max_seq_len());
  if (input_ids.size() > capacity) throw std::length_error("Prompt exceeds speculative context capacity");
  const size_t limit = std::min(max_length, capacity);
  if (limit <= input_ids.size()) return;
  for (auto token : input_ids) validate_speculative_token(token, target_model_->get_vocab_size());
  SpeculativeDeviceScope device_scope(cuda_device_id_);
  target_kv_cache_.clear();
  draft_kv_cache_.clear();
  try {
    // A fresh request owns its complete prompt. Both models borrow the same
    // upload until their synchronous prefill boundaries have completed.
    auto prompt = Tensor<uint32_t>::from_external_buffer(
        prompt_storage_.template ptr_at<uint32_t>(0), {input_ids.size()}, Device::CUDA);
    checkCudaErrors(cudaMemcpyAsync(prompt.data_ptr(), input_ids.data(), input_ids.size() * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice, verify_stream_));
    checkCudaErrors(cudaStreamSynchronize(verify_stream_));
    target_kv_cache_.resize(input_ids.size());
    auto target_logits = target_spec_model_->speculative_prefill_logits(&prompt, &target_kv_cache_);
    if (target_logits.shape[0] != input_ids.size())
      throw std::runtime_error("Target prefill returned an invalid extent");
    op::cuda::sample<T>(verify_context_,
                       target_logits.subview({input_ids.size() - 1, 0}, {1, target_logits.shape[1]}).as_const(),
                       {target_tokens_, {1}, {1}}, {}, target_scratch_, target_sampling_plan_,
                       temperature, top_p, top_k, target_states_);
    uint32_t pending = 0;
    checkCudaErrors(cudaMemcpyAsync(&pending, target_tokens_, sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost, verify_stream_));
    checkCudaErrors(cudaStreamSynchronize(verify_stream_));
    validate_speculative_token(pending, target_model_->get_vocab_size());
    callback(pending);
    if (pending == target_model_->get_eos_token_id()) return;
    size_t total_length = input_ids.size() + 1;
    if (total_length >= limit) return;
    draft_kv_cache_.resize(input_ids.size());
    draft_spec_model_->speculative_prefill_logits(&prompt, &draft_kv_cache_);
    std::vector<uint32_t> verified;
    verified.reserve(MAX_SPEC_LENGTH);
    while (total_length < limit) {
      const size_t original = target_kv_cache_.size();
      if (draft_kv_cache_.size() != original)
        throw std::logic_error("Speculative sessions lost their shared pending-token boundary");
      const size_t count = std::min({adaptive_spec_length_, limit - total_length, capacity - original});
      if (!count) break;
      checkCudaErrors(cudaMemcpyAsync(draft_tokens_, &pending, sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, verify_stream_));
      checkCudaErrors(cudaStreamSynchronize(verify_stream_));
      generate_draft_tokens(count, use_probability_ratio_, temperature, top_p, top_k);
      const size_t accepted = use_probability_ratio_
          ? verify_probability_ratio(count, temperature, top_p, top_k, verified)
          : verify_greedy(count, temperature, top_p, verified);
      if (verified.empty()) throw std::logic_error("Speculative verification must make progress");
      update_adaptive_spec_length(static_cast<float>(accepted) / count);
      size_t emitted = 0;
      bool eos = false;
      for (auto token : verified) {
        validate_speculative_token(token, target_model_->get_vocab_size());
        pending = token;
        callback(token);
        ++emitted;
        ++total_length;
        if (token == target_model_->get_eos_token_id()) { eos = true; break; }
      }
      // If EOS truncates an otherwise accepted batch, discard the unreported tail.
      target_kv_cache_.resize(original + emitted);
      draft_kv_cache_.resize(original + emitted);
      if (eos) break;
    }
  } catch (...) {
    // Propagate model and callback failures after restoring a reusable request boundary.
    cudaDeviceSynchronize();
    target_kv_cache_.clear();
    draft_kv_cache_.clear();
    throw;
  }
}

template class SpeculativeDecoder<__nv_bfloat16>;
