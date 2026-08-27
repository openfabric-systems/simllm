#ifndef SIMLLM_TRAF70_SHA256_H_
#define SIMLLM_TRAF70_SHA256_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace traf70 {

class Sha256 {
 public:
  Sha256() { reset(); }

  void reset() {
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    buffer_size_ = 0;
    total_bytes_ = 0;
  }

  void update(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    total_bytes_ += size;
    while (size > 0) {
      const std::size_t take = std::min(size, buffer_.size() - buffer_size_);
      for (std::size_t i = 0; i < take; ++i) buffer_[buffer_size_ + i] = bytes[i];
      buffer_size_ += take;
      bytes += take;
      size -= take;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_.data());
        buffer_size_ = 0;
      }
    }
  }

  std::string finish_hex() {
    const std::uint64_t bit_count = static_cast<std::uint64_t>(total_bytes_) * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56) {
      while (buffer_size_ < 64) buffer_[buffer_size_++] = 0;
      transform(buffer_.data());
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56) buffer_[buffer_size_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer_[buffer_size_++] = static_cast<unsigned char>(bit_count >> shift);
    }
    transform(buffer_.data());
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto value : state_) output << std::setw(8) << value;
    const auto result = output.str();
    reset();
    return result;
  }

 private:
  static std::uint32_t rotate_right(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
  }

  void transform(const unsigned char* block) {
    static constexpr std::array<std::uint32_t, 64> constants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t offset = i * 4;
      words[i] = (static_cast<std::uint32_t>(block[offset]) << 24) |
                 (static_cast<std::uint32_t>(block[offset + 1]) << 16) |
                 (static_cast<std::uint32_t>(block[offset + 2]) << 8) |
                 static_cast<std::uint32_t>(block[offset + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                               rotate_right(words[i - 15], 18) ^
                               (words[i - 15] >> 3);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                               rotate_right(words[i - 2], 19) ^
                               (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                 rotate_right(e, 25);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 = h + sum1 + choose + constants[i] + words[i];
      const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                 rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<unsigned char, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::size_t total_bytes_ = 0;
};

inline std::string sha256_hex(const void* data, std::size_t size) {
  Sha256 digest;
  digest.update(data, size);
  return digest.finish_hex();
}

}  // namespace traf70

#endif  // SIMLLM_TRAF70_SHA256_H_
