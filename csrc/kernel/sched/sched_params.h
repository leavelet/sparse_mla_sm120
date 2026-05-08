#pragma once

#include <cstdint>

// FlashMLA-compatible scheduling metadata for split-KV decode.
// Binary layout must match FlashMLA's DecodingSchedMeta exactly.

struct __align__(32) DecodingSchedMeta {
    int begin_req_idx;
    int end_req_idx;
    int begin_block_idx;
    int end_block_idx;
    int begin_split_idx;
    int is_first_req_splitted;
    int is_last_req_splitted;
    int _pad;
};
static_assert(sizeof(DecodingSchedMeta) == 32);
