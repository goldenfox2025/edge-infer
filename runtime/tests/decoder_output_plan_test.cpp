#include "execution/decoder.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void output_plan_test() {
  Qwen3Config config{};
  config.hidden_size = 1024;
  config.n_heads = 16;
  config.n_kv_heads = 8;
  config.head_dim = 128;
  config.intermediate_size = 3072;
  config.vocab_size = 151936;
  config.max_position_embeddings = 4096;
  for (const size_t rows : {size_t{1}, size_t{16}, size_t{1024}}) {
    const auto hidden = plan_decoder_workspace<__nv_bfloat16>(config, rows, DecoderOutput::Hidden);
    const auto tokens = plan_decoder_workspace<__nv_bfloat16>(config, rows);
    require(!hidden.has_allocation("logits") && tokens.has_allocation("logits"),
            "Only token output plans may request vocabulary logits");
    const size_t head_bytes = rows * config.vocab_size * sizeof(__nv_bfloat16);
    require(tokens.at("logits").bytes == head_bytes &&
                tokens.requested_bytes() - hidden.requested_bytes() == head_bytes,
            "Hidden output planning must omit exactly the unnecessary head request");
    require(hidden.total_bytes() < tokens.total_bytes(),
            "The large vocabulary head must not dominate a hidden-only arena");
    require(hidden.at("final_norm").bytes == rows * config.hidden_size * sizeof(__nv_bfloat16),
            "Hidden output planning must retain the normalized hidden output");
    require(hidden.at("attention_scratch").bytes == tokens.at("attention_scratch").bytes,
            "Choosing a head must not change the attention computation contract");
    std::cout << "rows=" << rows << ", hidden workspace=" << hidden.total_bytes()
              << ", token workspace=" << tokens.total_bytes()
              << ", omitted head request=" << head_bytes << '\n';
  }
}
}  // namespace

int main() {
  try {
    output_plan_test();
  } catch (const std::exception& error) {
    std::cerr << "decoder_output_plan_test failed: " << error.what() << '\n';
    return 1;
  }
}
