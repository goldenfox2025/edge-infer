#include "speech/qwen_tts_conditioning.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& file, std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error("Reference tensor byte extent overflow");
  }
  if (std::filesystem::file_size(file) != count * sizeof(T)) {
    throw std::runtime_error("Reference tensor size mismatch: " + file.string());
  }
  std::vector<T> result(count);
  std::ifstream input(file, std::ios::binary);
  input.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
  if (!input) throw std::runtime_error("Cannot read reference tensor: " + file.string());
  return result;
}

template <typename T> T encode(float value) { return static_cast<T>(value); }
template <> __nv_bfloat16 encode(float value) { return __float2bfloat16(value); }
template <typename T> float decode(T value) { return static_cast<float>(value); }
template <> float decode(__nv_bfloat16 value) { return __bfloat162float(value); }

int bf16_ordered(float value) {
  const auto rounded = __float2bfloat16(value);
  uint16_t bits;
  static_assert(sizeof(rounded) == sizeof(bits));
  std::memcpy(&bits, &rounded, sizeof(bits));
  return bits & 0x8000 ? 0x8000 - (bits & 0x7fff) : 0x8000 + bits;
}

template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t count) : count_(count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
  }
  ~Buffer() { cudaFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  op::ArrayView<T> view() { return {data_, count_}; }
  op::ArrayView<const T> input() const { return {data_, count_}; }
  void upload(const std::vector<T>& data, cudaStream_t stream) {
    if (data.size() != count_) throw std::runtime_error("Upload extent mismatch");
    cuda_check(cudaMemcpyAsync(data_, data.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaStreamSynchronize(stream));
  }
  std::vector<T> download(cudaStream_t stream) const {
    std::vector<T> result(count_);
    cuda_check(cudaMemcpyAsync(result.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    return result;
  }
 private:
  T* data_ = nullptr;
  std::size_t count_;
};

struct Context {
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  Context() {
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("Cannot create cuBLAS handle");
    }
  }
  ~Context() { cublasDestroy(handle); cudaStreamDestroy(stream); }
};

template <typename T>
void upload_float(Buffer<T>& buffer, const std::filesystem::path& file,
                  std::size_t count, cudaStream_t stream) {
  auto values = read_binary<float>(file, count);
  std::vector<T> converted;
  converted.reserve(count);
  for (float value : values) converted.push_back(encode<T>(value));
  buffer.upload(converted, stream);
}

template <typename T>
std::vector<T> rotate_rows(const std::vector<T>& values, std::size_t rows,
                           std::size_t width) {
  if (rows < 2 || values.size() != rows * width) {
    throw std::runtime_error("Caller-state isolation requires at least two complete fixture rows");
  }
  std::vector<T> rotated(values.size());
  for (std::size_t row = 0; row < rows; ++row) {
    std::copy_n(values.data() + ((row + 1) % rows) * width, width,
                rotated.data() + row * width);
  }
  return rotated;
}

template <typename T>
void compare(const std::vector<T>& actual, const std::vector<float>& expected,
             const char* phase, bool bf16) {
  float maximum = 0;
  int maximum_steps = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float value = decode(actual[i]);
    const float error = std::abs(value - expected[i]);
    maximum = std::max(maximum, error);
    const auto steps = bf16 ? std::abs(bf16_ordered(value) - bf16_ordered(expected[i])) : 0;
    maximum_steps = std::max(maximum_steps, steps);
    if (!std::isfinite(value) || !std::isfinite(expected[i]) ||
        (bf16 ? steps > 1 : error > 1e-6f + 1e-5f * std::abs(expected[i]))) {
      throw std::runtime_error(std::string(phase) + " mismatch at " + std::to_string(i) +
                               ": " + std::to_string(value) + " vs " + std::to_string(expected[i]));
    }
  }
  std::cout << phase << ": max absolute error " << maximum << '\n';
  if (bf16) std::cout << phase << ": max BF16 representable steps " << maximum_steps << '\n';
}

template <typename T>
void run(const std::filesystem::path& directory, bool bf16) {
  std::size_t text, intermediate, hidden, groups, rows, vocabulary;
  std::ifstream dimensions(directory / "dimensions.txt");
  if (!(dimensions >> text >> intermediate >> hidden >> groups >> rows >> vocabulary) ||
      text == 0 || intermediate == 0 || hidden == 0 || groups == 0 || rows == 0 || vocabulary == 0 ||
      text > 4096 || intermediate > 4096 || hidden > 4096 || groups > 32 || rows > 64 || vocabulary > 64) {
    throw std::runtime_error("Invalid conditioning fixture dimensions");
  }
  Context context;
  Buffer<T> w1(text * intermediate), b1(intermediate), w2(hidden * intermediate), b2(hidden);
  Buffer<T> input(rows * text), projected(rows * hidden), middle(rows * intermediate), composed(rows * hidden);
  Buffer<float> scratch(rows * std::max(intermediate, hidden));
  upload_float(w1, directory / "fc1_weight.f32", text * intermediate, context.stream);
  upload_float(b1, directory / "fc1_bias.f32", intermediate, context.stream);
  upload_float(w2, directory / "fc2_weight.f32", hidden * intermediate, context.stream);
  upload_float(b2, directory / "fc2_bias.f32", hidden, context.stream);
  upload_float(input, directory / "text_input.f32", rows * text, context.stream);
  std::vector<std::unique_ptr<Buffer<T>>> table_storage;
  std::vector<op::cuda::EmbeddingTable<T>> tables;
  for (std::size_t group = 0; group < groups; ++group) {
    auto storage = std::make_unique<Buffer<T>>(vocabulary * hidden);
    upload_float(*storage, directory / ("codec_" + std::to_string(group) + ".f32"), vocabulary * hidden, context.stream);
    tables.push_back({storage->input(), vocabulary});
    table_storage.push_back(std::move(storage));
  }
  const edge_infer::speech::QwenTtsConditioner<T> stage(
      {text, intermediate, hidden, groups},
      {w1.input(), b1.input(), w2.input(), b2.input(), {tables.data(), tables.size()}});
  stage.project_text(input.input(), projected.view(), middle.view(), scratch.view(),
                     rows, context.handle, context.stream);
  const std::string dtype = bf16 ? "bfloat16" : "float32";
  compare(projected.download(context.stream),
          read_binary<float>(directory / ("text_expected_" + dtype + ".f32"), rows * hidden),
          (dtype + " text projection").c_str(), bf16);
  auto codes = read_binary<uint32_t>(directory / "codec_ids.u32", rows * groups);
  stage.compose_frame_embeddings({codes.data(), codes.size()}, projected.input(),
                                 composed.view(), scratch.view(), rows, context.stream);
  compare(composed.download(context.stream),
          read_binary<float>(directory / ("frame_expected_" + dtype + ".f32"), rows * hidden),
          (dtype + " codec + text").c_str(), bf16);
  bool rejected_alias = false;
  try {
    stage.compose_frame_embeddings({codes.data(), codes.size()}, composed.input(),
                                   composed.view(), scratch.view(), rows, context.stream);
  } catch (const std::invalid_argument&) { rejected_alias = true; }
  if (!rejected_alias) throw std::runtime_error("Composed frame overwrote aliased text");
  // The embedding sum writes scratch before it adds text. A text view into that
  // workspace would silently produce twice the codec sum without this guard.
  bool rejected_scratch_alias = false;
  try {
    stage.compose_frame_embeddings(
        {codes.data(), codes.size()},
        {reinterpret_cast<const T*>(scratch.view().data), rows * hidden},
        composed.view(), scratch.view(), rows, context.stream);
  } catch (const std::invalid_argument&) { rejected_scratch_alias = true; }
  if (!rejected_scratch_alias) throw std::runtime_error("Composed frame overwrote text in workspace");
  bool rejected_projection_alias = false;
  try {
    stage.project_text(input.input(), projected.view(),
                       {reinterpret_cast<T*>(scratch.view().data), rows * intermediate},
                       scratch.view(), rows, context.handle, context.stream);
  } catch (const std::invalid_argument&) { rejected_projection_alias = true; }
  if (!rejected_projection_alias) throw std::runtime_error("Projection used GEMM input as workspace");

  // The immutable stage borrows the same weights for both callers. Every
  // mutable buffer, stream and cuBLAS handle belongs to one caller only.
  Context second_context;
  Buffer<T> second_input(rows * text), second_projected(rows * hidden);
  Buffer<T> second_middle(rows * intermediate), second_composed(rows * hidden);
  Buffer<float> second_scratch(rows * std::max(intermediate, hidden));
  const auto original_input = read_binary<float>(directory / "text_input.f32", rows * text);
  const auto rotated_input = rotate_rows(original_input, rows, text);
  const auto rotated_codes = rotate_rows(codes, rows, groups);
  if (rotated_input == original_input || rotated_codes == codes) {
    throw std::runtime_error("Caller-state isolation requires different text inputs and codec rows");
  }
  std::vector<T> second_input_values;
  second_input_values.reserve(rotated_input.size());
  for (float value : rotated_input) second_input_values.push_back(encode<T>(value));
  second_input.upload(second_input_values, second_context.stream);
  const auto text_reference = read_binary<float>(
      directory / ("text_expected_" + dtype + ".f32"), rows * hidden);
  const auto frame_reference = read_binary<float>(
      directory / ("frame_expected_" + dtype + ".f32"), rows * hidden);
  const auto rotated_text_reference = rotate_rows(text_reference, rows, hidden);
  const auto rotated_frame_reference = rotate_rows(frame_reference, rows, hidden);

  // Enqueue both callers before any readback/synchronization. Each caller
  // reuses its preallocated scratch sequentially on its own stream.
  stage.project_text(input.input(), projected.view(), middle.view(), scratch.view(),
                     rows, context.handle, context.stream);
  stage.project_text(second_input.input(), second_projected.view(), second_middle.view(),
                     second_scratch.view(), rows, second_context.handle, second_context.stream);
  stage.compose_frame_embeddings({codes.data(), codes.size()}, projected.input(),
                                 composed.view(), scratch.view(), rows, context.stream);
  stage.compose_frame_embeddings({rotated_codes.data(), rotated_codes.size()},
                                 second_projected.input(), second_composed.view(),
                                 second_scratch.view(), rows, second_context.stream);
  compare(projected.download(context.stream), text_reference,
          (dtype + " caller 1 text projection").c_str(), bf16);
  compare(composed.download(context.stream), frame_reference,
          (dtype + " caller 1 codec + text").c_str(), bf16);
  compare(second_projected.download(second_context.stream), rotated_text_reference,
          (dtype + " caller 2 rotated text projection").c_str(), bf16);
  compare(second_composed.download(second_context.stream), rotated_frame_reference,
          (dtype + " caller 2 rotated codec + text").c_str(), bf16);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "Usage: qwen_tts_conditioning_parity <fixture-directory>\n"; return 2; }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  try {
    run<float>(argv[1], false);
    run<__nv_bfloat16>(argv[1], true);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
