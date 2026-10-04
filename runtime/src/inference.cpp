#include "inference.hpp"

#include <chrono>
#include <iomanip>
#include <numeric>
#include <limits>
#include <vector>

#include "base_model.hpp"
#include "common.hpp"
#include "operators/unified_operators.hpp"
#include "qwen.hpp"

enum class Signal { EndOfStream };
using GenerationResult = std::variant<uint32_t, Signal, std::exception_ptr>;

namespace {

class DeviceTimer {
 public:
  explicit DeviceTimer(Device device) : device_(device) {
    if (device_ == Device::CUDA) {
      gpu_timer_ = std::make_unique<GpuTimer>();
    }
  }

  void start(cudaStream_t stream = nullptr) {
    if (gpu_timer_) {
      gpu_timer_->start(stream);
    } else {
      cpu_timer_.start();
    }
  }

  void stop(cudaStream_t stream = nullptr) {
    if (gpu_timer_) {
      gpu_timer_->stop(stream);
    } else {
      cpu_timer_.stop();
    }
  }

  float milliseconds() {
    return gpu_timer_ ? gpu_timer_->milliseconds()
                      : static_cast<float>(cpu_timer_.milliseconds());
  }

 private:
  Device device_;
  std::unique_ptr<GpuTimer> gpu_timer_;
  CpuTimer cpu_timer_;
};

uint32_t read_token_from_device(uint32_t* token_ptr, Device device) {
  if (device == Device::CUDA) {
    uint32_t token = 0;
    checkCudaErrors(cudaMemcpyAsync(&token, token_ptr, sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost,
                                    cudaStreamDefault));
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    return token;
  }

  if (!token_ptr) {
    throw std::runtime_error("Received null CPU token pointer");
  }
  return *token_ptr;
}

void validate_token_id(const BaseModel* model, uint32_t token,
                       const char* phase) {
  if (!model) {
    throw std::runtime_error("Model is null while validating token id");
  }

  const size_t vocab_size = model->get_vocab_size();
  if (token >= vocab_size) {
    throw std::runtime_error(std::string("Invalid token id from ") + phase +
                             ": " + std::to_string(token) +
                             " >= vocab_size " +
                             std::to_string(vocab_size));
  }
}

size_t estimate_prefill_arena_bytes(const BaseModel* model, size_t token_count,
                                    size_t element_size) {
  if (model == nullptr || token_count == 0) {
    return 0;
  }

  const size_t model_estimate = model->estimate_prefill_workspace_bytes(token_count);
  if (model_estimate > 0) {
    const size_t slack_bytes =
        std::max<size_t>(32 * 1024 * 1024, model_estimate / 8);
    return model_estimate + slack_bytes;
  }

  const size_t hidden = model->get_hidden_size();
  const size_t kv_width = model->get_n_kv_heads() * model->get_head_dim();
  const size_t per_layer_elements = (6 * hidden) + (4 * kv_width);
  const size_t total_elements =
      token_count * model->get_n_layers() * per_layer_elements;
  const size_t estimated_bytes = total_elements * element_size;
  const size_t slack_bytes = std::max<size_t>(32 * 1024 * 1024,
                                              estimated_bytes / 4);
  return estimated_bytes + slack_bytes;
}

class LegacyPrefillPhase {
 public:
  LegacyPrefillPhase(bool enabled, const BaseModel* model, size_t tokens,
                     size_t element_size) : active_(enabled) {
    if (active_) {
      GlobalCudaMemoryPool::prepare_prefill_capacity(
          estimate_prefill_arena_bytes(model, tokens, element_size));
      GlobalCudaMemoryPool::set_prefill_phase(true);
    }
  }
  ~LegacyPrefillPhase() { finish(); }
  void finish() {
    if (active_) {
      GlobalCudaMemoryPool::set_prefill_phase(false);
      active_ = false;
    }
  }
  LegacyPrefillPhase(const LegacyPrefillPhase&) = delete;
  LegacyPrefillPhase& operator=(const LegacyPrefillPhase&) = delete;
 private:
  bool active_;
};

}  // namespace

template <typename T>
KVCache<T>::KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim, Device device, size_t initial_size)
    : n_layers_(n_layers), max_seq_len_(max_seq_len), head_dim_(head_dim), current_len_(0), device_(device) {

    if (!n_layers_ || !max_seq_len_ || !head_dim_ ||
        n_layers_ > std::numeric_limits<size_t>::max() / max_seq_len_ ||
        n_layers_ * max_seq_len_ > std::numeric_limits<size_t>::max() / head_dim_ / sizeof(T)) {
        throw std::invalid_argument("KVCache dimensions are empty or exceed addressable storage");
    }
    if (initial_size > max_seq_len_) {
        throw std::invalid_argument("Initial size cannot exceed max_seq_len");
    }
    const std::vector<size_t> shape{n_layers_, max_seq_len_, head_dim_};
    if (device_ == Device::CUDA) {
        const size_t bytes = n_layers_ * max_seq_len_ * head_dim_ * sizeof(T);
        k_storage_.reserve(bytes);
        v_storage_.reserve(bytes);
        k_cache_contiguous_ = Tensor<T>::from_external_buffer(k_storage_.template ptr_at<T>(0), shape, device_);
        v_cache_contiguous_ = Tensor<T>::from_external_buffer(v_storage_.template ptr_at<T>(0), shape, device_);
    } else {
        k_cache_contiguous_ = Tensor<T>(shape, device_);
        v_cache_contiguous_ = Tensor<T>(shape, device_);
    }

    k_cache_slices_.resize(n_layers_ * max_seq_len_);
    v_cache_slices_.resize(n_layers_ * max_seq_len_);
    for (size_t layer = 0; layer < n_layers_; layer++) {
        for (size_t pos = 0; pos < max_seq_len_; pos++) {
            size_t idx = layer * max_seq_len_ + pos;

            k_cache_slices_[idx] = k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
            v_cache_slices_[idx] = v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
        }
    }

    if (initial_size > 0) {
        if (initial_size > max_seq_len_) {
            throw std::runtime_error("Initial size cannot exceed max_seq_len");
        }
        current_len_ = initial_size;
    }
}

template <typename T>
void KVCache<T>::resize(size_t new_size) {
    if (new_size > max_seq_len_) {
        throw std::runtime_error("KVCache: Attempted to resize beyond max_seq_len");
    }
    current_len_ = new_size;
}

template <typename T>
void KVCache<T>::clear() {
    current_len_ = 0;
}
template <typename T>
Tensor<T>& KVCache<T>::k_cache(size_t layer, size_t pos) {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range");
    }
    if (pos >= max_seq_len_) {
        throw std::runtime_error("KVCache: Position index out of range");
    }
    size_t idx = layer * max_seq_len_ + pos;
    return k_cache_slices_[idx];
}

template <typename T>
Tensor<T>& KVCache<T>::v_cache(size_t layer, size_t pos) {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range");
    }
    if (pos >= max_seq_len_) {
        throw std::runtime_error("KVCache: Position index out of range");
    }
    size_t idx = layer * max_seq_len_ + pos;

    return v_cache_slices_[idx];
}

template <typename T>
KVCache<T>& KVCache<T>::cuda() {
    if (device_ == Device::CUDA)
        return *this;

    // Tensor::nbytes() is a legacy int API. Persistent cache capacities can
    // exceed 2 GiB per buffer, so compute allocation/copy sizes in size_t.
    const size_t k_bytes = k_cache_contiguous_.numel() * sizeof(T);
    const size_t v_bytes = v_cache_contiguous_.numel() * sizeof(T);
    k_storage_.reserve(k_bytes);
    v_storage_.reserve(v_bytes);
    auto next_k = Tensor<T>::from_external_buffer(k_storage_.template ptr_at<T>(0),
                                                 k_cache_contiguous_.sizes(), Device::CUDA);
    auto next_v = Tensor<T>::from_external_buffer(v_storage_.template ptr_at<T>(0),
                                                 v_cache_contiguous_.sizes(), Device::CUDA);
    checkCudaErrors(cudaMemcpy(next_k.data_ptr(), k_cache_contiguous_.data_ptr(),
                               k_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(next_v.data_ptr(), v_cache_contiguous_.data_ptr(),
                               v_bytes, cudaMemcpyHostToDevice));
    k_cache_contiguous_ = std::move(next_k);
    v_cache_contiguous_ = std::move(next_v);
    device_ = Device::CUDA;

    // Rebuild views after the underlying storage pointer changes.
    for (size_t layer = 0; layer < n_layers_; layer++) {
        for (size_t pos = 0; pos < max_seq_len_; pos++) {
            size_t idx = layer * max_seq_len_ + pos;
            k_cache_slices_[idx] = k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
            v_cache_slices_[idx] = v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
        }
    }
    return *this;
}

template <typename T>
KVCache<T>& KVCache<T>::cpu() {
    if (device_ == Device::CPU)
        return *this;

    device_ = Device::CPU;

    k_cache_contiguous_ = k_cache_contiguous_.cpu();
    v_cache_contiguous_ = v_cache_contiguous_.cpu();
    k_storage_.release();
    v_storage_.release();

    // Rebuild views after the underlying storage pointer changes.
    for (size_t layer = 0; layer < n_layers_; layer++) {
        for (size_t pos = 0; pos < max_seq_len_; pos++) {
            size_t idx = layer * max_seq_len_ + pos;
            k_cache_slices_[idx] = k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
            v_cache_slices_[idx] = v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
        }
    }
    return *this;
}

template <typename T>
std::pair<const Tensor<T>, const Tensor<T>> KVCache<T>::get_contiguous_tensor(size_t layer) const {
    Tensor<T> K = k_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_}).squeeze(0);
    Tensor<T> V = v_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_}).squeeze(0);

    return {K, V};
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> KVCache<T>::get_layer_view(size_t layer) {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range in get_layer_view");
    }
    // Return a view (slice) of the specified layer.
    // The view is writable and points to the continuous memory block of that layer.
    Tensor<T> k_view = k_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, max_seq_len_, head_dim_}).squeeze(0);
    Tensor<T> v_view = v_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, max_seq_len_, head_dim_}).squeeze(0);
    return {k_view, v_view};
}

template class KVCache<float>;
template class KVCache<__nv_bfloat16>;

// ------------------------

// ------------------------

template <typename T>
bool InferenceEngine<T>::has_warmed_up_ = false;
template <typename T>
InferenceEngine<T>::InferenceEngine(std::shared_ptr<BaseModel> model, Device device)
    : model_(model),

      kv_cache_(model_->get_n_layers(), model_->get_max_seq_len(), model_->get_head_dim() * model_->get_n_kv_heads(),
                device),
      thread_pool_(4),
      device_(device),
      d_states(nullptr),
      operators_(std::make_unique<op::UnifiedOperators<T>>(device)),
      benchmark_mode_(true),
      benchmark_warmup_tokens_(64) {

    if (device_ == Device::CUDA) {

        if (model_->device() != Device::CUDA) {
            model_->cuda();
        }

        cudaError_t err = cudaMalloc(&d_states, sizeof(curandState));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory for curand states: " +
                                     std::string(cudaGetErrorString(err)));
        }
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        operators_->cuda();
        operators_->init_curand(d_states, seed, 0, nullptr);
        // Nonblocking session streams must observe initialized RNG state.
        checkCudaErrors(cudaStreamSynchronize(nullptr));
        this->cuda();
    }

    Device model_device = model_->device();
    std::cout << "[Engine initialized] Model device: " << (model_device == Device::CUDA ? "CUDA" : "CPU") << std::endl;
    std::cout << "[Engine initialized] Engine device: " << (device_ == Device::CUDA ? "CUDA" : "CPU") << std::endl;
}

template <typename T>
InferenceEngine<T>::~InferenceEngine() {

    if (d_states != nullptr && device_ == Device::CUDA) {
        // Wait for outstanding CUDA work before releasing resources.
        cudaDeviceSynchronize();
        cudaFree(d_states);
        d_states = nullptr;
    }
}

template <typename T>
uint32_t* InferenceEngine<T>::generate_next_token(ThreadPool& thread_pool, uint32_t* input_ids, float temperature,
                                                  float top_p, size_t top_k) {
    DeviceTimer token_gen_timer(device_);
    token_gen_timer.start();

    Tensor<uint32_t> input;
    if (device_ == Device::CUDA) {
        input = Tensor<uint32_t>(input_ids, {1}, device_);
    } else {
        input = Tensor<uint32_t>(std::vector<uint32_t>{*input_ids}, {1}, Device::CPU);
    }

    try {
        kv_cache_.resize(kv_cache_.size() + 1);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error resizing KV cache: " << e.what() << std::endl;
        throw;
    }

    uint32_t* next_token;

    if (device_ == Device::CUDA) {
        next_token = model_->forward(&input, thread_pool, &kv_cache_, top_k, temperature, top_p, d_states);
    } else {
        next_token = model_->forward(&input, thread_pool, &kv_cache_, top_k, temperature, top_p);
    }

    token_gen_timer.stop();

    return next_token;
}

template <typename T>
void InferenceEngine<T>::warmup(size_t warmup_tokens, bool force_warmup, float temperature, float top_p, size_t top_k) {

    if ((!has_warmed_up_ || force_warmup) && device_ == Device::CUDA) {
        std::cout << "Running CUDA warmup (tokens: " << warmup_tokens << ")..." << std::endl << std::flush;

        std::vector<uint32_t> warmup_input(warmup_tokens, 1);

        size_t original_kv_size = kv_cache_.size();

        try {

            GpuTimer warmup_timer;
            warmup_timer.start();

            LegacyPrefillPhase prefill_phase(!model_->owns_execution_workspaces(),
                                             model_.get(), warmup_input.size(), sizeof(T));

            kv_cache_.resize(warmup_input.size());

            std::vector<uint32_t> warmup_input_copy = warmup_input;
            Tensor<uint32_t> input_tensor(std::move(warmup_input_copy), {warmup_input.size()}, device_);

            uint32_t* warmup_token =
                model_->prefill(&input_tensor, thread_pool_, &kv_cache_, top_k, temperature, top_p, d_states);
            prefill_phase.finish();

            // Initialize the Qwen CUDA graph through a decode call during warmup.
            // This lets graph capture use the active KV cache.
            std::cout << "Checking whether CUDA graph initialization is needed..." << std::endl;

            auto qwen_model_bf16 = dynamic_cast<QwenModel<__nv_bfloat16>*>(model_.get());
            auto qwen_model_float = dynamic_cast<QwenModel<float>*>(model_.get());

            if (qwen_model_bf16 || qwen_model_float) {
                std::cout << "QwenModel detected; initializing CUDA graph..." << std::endl;
                try {

                    kv_cache_.resize(1);

                    std::vector<uint32_t> graph_init_input = {9707};
                    Tensor<uint32_t> graph_input_tensor_(std::move(graph_init_input), {1}, device_);

                    uint32_t* graph_warmup_token = model_->forward(&graph_input_tensor_, thread_pool_, &kv_cache_,
                                                                   top_k, temperature, top_p, d_states);

                    std::cout << "CUDA graph initialization completed." << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "CUDA graph initialization skipped or failed: " << e.what() << std::endl;

                }
            }

            warmup_timer.stop();

            kv_cache_.clear();

            if (!model_->owns_execution_workspaces()) GlobalCudaMemoryPool::reset_prefill_buffer();

            if (!force_warmup) {
                has_warmed_up_ = true;
            }
            kv_cache_.resize(0);
            std::cout << "CUDA warmup completed in: " << std::fixed << std::setprecision(2) << warmup_timer.milliseconds()
                      << " ms" << std::endl
                      << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "Error during warmup: " << e.what() << std::endl;
            // A warmup failure does not prevent normal inference.
        }
    } else if (device_ == Device::CPU) {
        std::cout << "CPU mode; skipping CUDA warmup" << std::endl;
    } else if (has_warmed_up_ && !force_warmup) {
        std::cout << "Warmup already completed; skipping repeated warmup" << std::endl;
    }
}

template <typename T>
void InferenceEngine<T>::set_benchmark_mode(bool enable_benchmark, size_t benchmark_warmup_tokens) {
    benchmark_mode_ = enable_benchmark;
    benchmark_warmup_tokens_ = benchmark_warmup_tokens;

    if (enable_benchmark) {
        std::cout << "Benchmark mode enabled; warmup before each inference: " << benchmark_warmup_tokens << " tokens" << std::endl;
    } else {
        std::cout << "Benchmark mode disabled" << std::endl;
    }
}

template <typename T>
void InferenceEngine<T>::generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                                                float temperature, float top_p, size_t top_k,
                                                std::function<void(uint32_t)> callback) {

    if (benchmark_mode_ && device_ == Device::CUDA) {
        std::cout << "Benchmark mode: starting warmup..." << std::endl;
        warmup(input_ids.size(), true, temperature, top_p, top_k);
    }

    else if (!has_warmed_up_ && device_ == Device::CUDA) {
        warmup(64, false, temperature, top_p, top_k);
    }

    ThreadSafeQueue<GenerationResult> result_queue;
    bind_this_thread_to_core(3);
    std::thread generation_thread([&, this, input_ids_copy = input_ids]() {
        try {
            uint32_t* next_token_ptr;
            uint32_t next_token_host = -1;
            size_t input_size = input_ids_copy.size();

            DeviceTimer total_prefill_timer(this->device_);
            total_prefill_timer.start();

            {
                LegacyPrefillPhase prefill_phase(
                    this->device_ == Device::CUDA && !this->model_->owns_execution_workspaces(),
                    this->model_.get(), input_size, sizeof(T));
                std::cerr << "Entering prefill; sequence length: " << input_size << std::endl;

                DeviceTimer prefill_timer(this->device_);
                prefill_timer.start();

                kv_cache_.resize(kv_cache_.size() + input_size);
                std::vector<uint32_t> prefill_input = input_ids_copy;
                Tensor<uint32_t> input_tensor(std::move(prefill_input), {input_size}, this->device_);
                next_token_ptr = this->model_->prefill(&input_tensor, this->thread_pool_, &this->kv_cache_, top_k,
                                                       temperature, top_p, this->d_states);

                prefill_timer.stop();

                prefill_phase.finish();

                std::cout << "Prefill completed in: " << std::fixed << std::setprecision(2)
                          << prefill_timer.milliseconds() << " ms" << std::endl
                          << std::flush;

                std::cerr << "Leaving prefill" << std::endl;
            }

            DeviceTimer token_timer(this->device_);
            token_timer.start();
            next_token_host = read_token_from_device(next_token_ptr, this->device_);
            validate_token_id(this->model_.get(), next_token_host, "prefill");

            token_timer.stop();

            std::cout << "First-token handling time: " << std::fixed << std::setprecision(2) << token_timer.milliseconds()
                      << " ms" << std::endl
                      << std::flush;

            total_prefill_timer.stop();
            std::cout << "Total prefill time: " << std::fixed << std::setprecision(2)
                      << total_prefill_timer.milliseconds() << " ms" << std::endl
                      << std::flush;

            if (next_token_host == this->model_->get_eos_token_id()) {
                if (this->device_ == Device::CPU && next_token_ptr != nullptr) {
                    delete next_token_ptr;
                }
                result_queue.push(Signal::EndOfStream);
                return;
            }

            result_queue.push(next_token_host);

            size_t current_total_length = input_size + 1;
            uint32_t* last_token_ptr = next_token_ptr;

            std::vector<float> decode_times;

            while (current_total_length < max_length) {
                DeviceTimer full_token_timer(this->device_);
                full_token_timer.start();

                next_token_ptr =
                    this->generate_next_token(this->thread_pool_, last_token_ptr, temperature, top_p, top_k);
                if (this->device_ == Device::CPU) {
                    delete last_token_ptr;
                }
                next_token_host = read_token_from_device(next_token_ptr, this->device_);
                validate_token_id(this->model_.get(), next_token_host, "decode");

                full_token_timer.stop();
                decode_times.push_back(full_token_timer.milliseconds());

                last_token_ptr = next_token_ptr;
                current_total_length++;
                bool is_eos = (next_token_host == this->model_->get_eos_token_id());

                if (is_eos) {
                    result_queue.push(Signal::EndOfStream);
                    break;
                }

                result_queue.push(next_token_host);
            }  // end while loop

            if (this->device_ == Device::CPU && last_token_ptr != nullptr) {
                delete last_token_ptr;
                last_token_ptr = nullptr;
            }

            result_queue.push(Signal::EndOfStream);

            if (!decode_times.empty()) {
                float total_decode_time = std::accumulate(decode_times.begin(), decode_times.end(), 0.0f);
                float average_decode_time = total_decode_time / decode_times.size();

                std::cout << "\n-------------------- Decode performance --------------------" << std::endl;
                std::cout << "Decoded tokens: " << decode_times.size() << "" << std::endl;
                std::cout << std::fixed << std::setprecision(2) << "Total decode time: " << total_decode_time << " ms"
                          << std::endl;
                std::cout << std::fixed << std::setprecision(2) << "Average decode time: " << average_decode_time
                          << " ms/token" << std::endl;
                std::cout << std::fixed << std::setprecision(2) << "Average decode rate: " << (1000.0f / average_decode_time)
                          << " tokens/s" << std::endl;
                std::cout << "--------------------------------------------------------" << std::endl << std::flush;
            }

        } catch (...) {
            // Forward worker exceptions to the result consumer.
            result_queue.push(std::current_exception());
        }
    });

    try {
        while (true) {
            GenerationResult result = result_queue.pop();
            bool should_break = false;
            std::visit(
                [&](auto&& arg) {
                    using Type = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<Type, uint32_t>) {
                        callback(arg);
                    } else if constexpr (std::is_same_v<Type, Signal>) {
                        if (arg == Signal::EndOfStream) {
                            should_break = true;
                        }
                    } else if constexpr (std::is_same_v<Type, std::exception_ptr>) {
                        if (arg) {
                            std::rethrow_exception(arg);  // Propagate the worker exception on the caller thread.
                        } else {

                            throw std::runtime_error("Worker thread sent null exception pointer.");
                        }
                    }
                },
                result);

            if (should_break) {
                break;
            }
        }
    } catch (...) {
        if (generation_thread.joinable()) {
            generation_thread.join();
        }
        throw;
    }

    // --- Cleanup ---
    // Join the worker on normal completion as well as on exceptions.
    if (generation_thread.joinable()) {
        generation_thread.join();
    }

    // Reset the prefill workspace cursor and retain its storage for reuse.

    if (device_ == Device::CUDA && !model_->owns_execution_workspaces()) {
        GlobalCudaMemoryPool::reset_prefill_buffer();
    }
}
template <typename T>
void InferenceEngine<T>::reset() {
    kv_cache_.clear();

    auto qwen_model_bf16 = dynamic_cast<QwenModel<__nv_bfloat16>*>(model_.get());
    auto qwen_model_float = dynamic_cast<QwenModel<float>*>(model_.get());
}
template <typename T>
InferenceEngine<T>& InferenceEngine<T>::cuda() {
    if (device_ == Device::CUDA) {
        return *this;
    }
    if (model_->device() == Device::CPU) {
        model_->cuda();
    }

    kv_cache_.cuda();
    if (!operators_) {
        operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
    } else {
        operators_->cuda();
    }
    if (d_states == nullptr) {
        cudaError_t err = cudaMalloc(&d_states, sizeof(curandState));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory for curand states: " +
                                     std::string(cudaGetErrorString(err)));
        }
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        operators_->init_curand(d_states, seed, 0, nullptr);
        checkCudaErrors(cudaStreamSynchronize(nullptr));
    }

    device_ = Device::CUDA;
    return *this;
}
template <typename T>
InferenceEngine<T>& InferenceEngine<T>::cpu() {
    if (device_ == Device::CPU) {
        return *this;
    }
    model_->cpu();
    kv_cache_.cpu();
    if (operators_) {
        operators_->cpu();
    }
    if (d_states != nullptr) {
        cudaDeviceSynchronize();
        cudaFree(d_states);
        d_states = nullptr;
    }
    device_ = Device::CPU;
    return *this;
}

template class InferenceEngine<float>;
template class InferenceEngine<__nv_bfloat16>;
