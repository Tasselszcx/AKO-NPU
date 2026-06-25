#ifndef ATTENTION_BACKWARD_TILING_H
#define ATTENTION_BACKWARD_TILING_H

#include <cstdint>

struct AttentionBackwardTilingData {
    // Basic shape parameters
    int32_t batchSize;
    int32_t seqLenQ;       // sq
    int32_t seqLenKV;      // skv
    int32_t numHeads;      // 80
    int32_t numKVHeads;    // 8
    int32_t numGroups;     // 10
    int32_t headDim;       // 128

    // Multi-core split parameters
    int32_t totalKVTasks;  // B * numKVHeads = total (b, kv_h) tasks
    int32_t tasksPerCore;  // ceil(totalKVTasks / usedCoreNum)
    int32_t usedCoreNum;

    // UB tiling parameters (Elementwise Phase)
    int32_t tileSq;        // rows per tile in softmax backward
    int32_t numSqTiles;    // number of sq tiles

    // UB tiling parameters (GQA Aggregation Phase)
    int32_t tileSkvAgg;    // rows per tile in GQA aggregation

    // Alignment parameters
    int32_t skvAlignedF32; // skv aligned to 8 (32B/4) for float32
    int32_t skvAlignedBf16;// skv aligned to 16 (32B/2) for bf16
    int32_t skvAlignedU8;  // skv aligned to 32 (32B/1) for uint8

    // Workspace offsets (per core)
    int64_t wsGradOutTOffset;    // offset to ws_grad_out_t region
    int64_t wsTempMM1Offset;     // offset to ws_temp_mm1 region
    int64_t wsTempMM2Offset;     // offset to ws_temp_mm2 region
    int64_t wsAccumOffset;       // offset to ws_accum region
    int64_t totalWorkspaceSize;  // total workspace bytes

    // Transpose buffer tile rows
    int32_t transposeTileRows;   // rows per tile when transposing grad_out

    // Constants
    float dropoutScale;    // 1.0 / (1.0 - 0.1) = 1.1111...

    // Per-launch group index (set by host for each kernel launch)
    int32_t currentG;      // which head group to process (0..numGroups-1)
};

#endif // ATTENTION_BACKWARD_TILING_H
