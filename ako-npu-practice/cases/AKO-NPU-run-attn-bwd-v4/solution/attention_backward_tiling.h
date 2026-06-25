#ifndef ATTENTION_BACKWARD_TILING_H
#define ATTENTION_BACKWARD_TILING_H

#include <cstdint>

struct AttentionBackwardTilingData {
    // Basic parameters
    uint32_t batchSize;
    uint32_t seqLenQ;
    uint32_t seqLenKV;
    uint32_t numHeads;         // = 80
    uint32_t numKVHeads;       // = 8
    uint32_t numGroups;        // = 10
    uint32_t headDim;          // = 128
    float dropoutScale;        // = 1.0 / (1.0 - 0.1) ~ 1.1111

    // Multi-core split
    uint32_t usedCores;
    uint32_t totalTasks;       // = batchSize * numKVHeads
    uint32_t tasksPerCore;

    // Vector tiling
    uint32_t tileRowsVec;      // Phase B: rows per tile
    uint32_t paddedSeqKV_f32;  // seq_kv padded for float32 (32-byte aligned)
    uint32_t paddedSeqKV_bf16; // seq_kv padded for bfloat16
    uint32_t paddedSeqKV_u8;   // seq_kv padded for uint8 (mask)

    // Workspace offsets (per core)
    // grad_out transpose: seq_q * headDim * 2 (bf16, 1 head)
    uint64_t wsGradOutOffset;
    // Matmul-1 output: seq_q * seq_kv * 4 (f32, 1 head)
    uint64_t wsMm1OutOffset;
    // Matmul-2 accumulation (f32): seq_kv * headDim * 4
    uint64_t wsGradVAccOffset;
    uint64_t wsPerCoreSize;        // total workspace per core
    uint64_t totalWorkspaceSize;   // total user workspace (all cores)
    uint64_t sysWorkspaceSize;     // system workspace size (for Matmul API)
};

#endif // ATTENTION_BACKWARD_TILING_H
