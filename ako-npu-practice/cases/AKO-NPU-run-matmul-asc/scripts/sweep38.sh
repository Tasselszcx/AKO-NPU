#!/bin/bash
# Iter 38 sweep: test different fixSplit + traverse + step combos with IterateAll
set -o pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
BACKUP="solution/matmul_leakyrelu.asc.backup"

# Source CANN env
[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh

RESULTS_FILE="sweep38_results.txt"
echo "=== Iter 38 Sweep Results ===" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

generate_asc() {
    local fixsplit_line="$1"
    local traverse="$2"
    local stepm="$3"
    local stepn="$4"

    cat > "$ASC_FILE" << 'ENDOFHEADER'
/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
*/
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);
    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;
    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                          GM_ADDR c, GM_ADDR workspace,
                                                                          const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);
    int32_t offsetA, offsetB, offsetC, offsetBias;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    matmulObj.template IterateAll<true>(cGlobal);
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                         int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                         int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;
    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

__global__ __mix__(1, 2) void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, AscendC::tiling::TCubeTiling tiling)
{
    AscendC::TPipe pipe;
    MatmulKernel<half, half, float, float> matmulKernel;
    matmulKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &matmulKernel.tiling);
    matmulKernel.Process(&pipe);

    uint32_t totalElements = tiling.M * tiling.N;
    uint32_t aivBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t aivBlockNum = 2;
    uint32_t elementsPerBlock = totalElements / aivBlockNum;
    uint32_t startElem = aivBlockIdx * elementsPerBlock;
    uint32_t endElem = (aivBlockIdx == aivBlockNum - 1) ? totalElements : startElem + elementsPerBlock;
    uint32_t myElements = endElem - startElem;

    const uint32_t CHUNK_SIZE = 12288;
    AscendC::GlobalTensor<float> cGm;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c) + startElem, myElements);

    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(inQueue, 2, CHUNK_SIZE * sizeof(float));
    pipe.InitBuffer(outQueue, 2, CHUNK_SIZE * sizeof(float));

    uint32_t numChunks = (myElements + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (uint32_t i = 0; i < numChunks; i++) {
        uint32_t offset = i * CHUNK_SIZE;
        uint32_t curSize = (myElements - offset < CHUNK_SIZE) ? (myElements - offset) : CHUNK_SIZE;
        uint32_t alignedSize = (curSize + 7) / 8 * 8;
        AscendC::LocalTensor<float> inBuf = inQueue.AllocTensor<float>();
        DataCopy(inBuf, cGm[offset], alignedSize);
        inQueue.EnQue(inBuf);
        AscendC::LocalTensor<float> computeBuf = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outBuf = outQueue.AllocTensor<float>();
        LeakyRelu(outBuf, computeBuf, (float)0.001, alignedSize);
        inQueue.FreeTensor(computeBuf);
        outQueue.EnQue(outBuf);
        AscendC::LocalTensor<float> writeBuf = outQueue.DeQue<float>();
        DataCopy(cGm[offset], writeBuf, alignedSize);
        outQueue.FreeTensor(writeBuf);
    }
}

ENDOFHEADER

    # Now write the tiling function with variable parameters
    cat >> "$ASC_FILE" << ENDOFTILING
AscendC::tiling::TCubeTiling GenerateTiling(platform_ascendc::PlatformAscendC* ascendcPlatform)
{
    using TPosition = matmul_tiling::TPosition;
    using CubeFormat = matmul_tiling::CubeFormat;
    using DataType = matmul_tiling::DataType;
    int M = 1024;
    int N = 640;
    int K = 256;
    int usedCoreNum = 2;

    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetMatmulConfigParams(1, true);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(true);
    tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::${traverse});
    ${fixsplit_line}
    tilingApi.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling tilingData;
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    tilingData.stepM = ${stepm};
    tilingData.stepN = ${stepn};
    return tilingData;
}
ENDOFTILING

    cat >> "$ASC_FILE" << 'ENDOFMAIN'

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = "Ascend910B1";
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    size_t aFileSize = 262144 * sizeof(int16_t);
    size_t bFileSize = 163840 * sizeof(int16_t);
    size_t cFileSize = 655360 * sizeof(float);
    size_t biasFileSize = 640 * sizeof(float);
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto tiling = GenerateTiling(ascendcPlatform);
    uint32_t numBlocks = 1;

    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t *inputAHost, *inputADevice;
    aclrtMallocHost((void **)(&inputAHost), aFileSize);
    aclrtMalloc((void **)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
    aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint8_t *inputBHost, *inputBDevice;
    aclrtMallocHost((void **)(&inputBHost), bFileSize);
    aclrtMalloc((void **)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
    aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint8_t *outputCHost, *outputCDevice;
    aclrtMallocHost((void **)(&outputCHost), cFileSize);
    aclrtMalloc((void **)&outputCDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    uint8_t *inputBiasHost, *inputBiasDevice;
    aclrtMallocHost((void **)(&inputBiasHost), biasFileSize);
    aclrtMalloc((void **)&inputBiasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
    aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint8_t *workspaceDevice;
    aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    matmul_leakyrelu_custom<<<numBlocks, nullptr, stream>>>(inputADevice, inputBDevice, inputBiasDevice, outputCDevice,
                                                           workspaceDevice, tiling);
    aclrtSynchronizeStream(stream);

    aclrtFree(inputADevice); aclrtFreeHost(inputAHost);
    aclrtFree(inputBDevice); aclrtFreeHost(inputBHost);
    aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output.bin", outputCHost, cFileSize);
    aclrtFree(outputCDevice); aclrtFreeHost(outputCHost);
    aclrtFree(inputBiasDevice); aclrtFreeHost(inputBiasHost);
    aclrtFree(workspaceDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
ENDOFMAIN
}

run_config() {
    local label="$1"
    local fixsplit_line="$2"
    local traverse="$3"
    local stepm="$4"
    local stepn="$5"

    echo "--- Testing: $label ---"

    # Generate the .asc file
    generate_asc "$fixsplit_line" "$traverse" "$stepm" "$stepn"

    # Build
    rm -rf solution/build
    cd solution
    export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake:${CMAKE_PREFIX_PATH:-}"
    mkdir -p build && chmod 755 build && cd build
    if ! cmake .. > /dev/null 2>&1 || ! make -j4 > /dev/null 2>&1; then
        echo "$label | COMPILE FAILED" | tee -a "../../$RESULTS_FILE"
        cd ../..
        return
    fi
    chmod 755 .

    # Gen data
    python3 ../scripts/gen_data.py > /dev/null 2>&1

    # Run
    rm -f output/output.bin
    timeout 30 ./demo > /dev/null 2>&1 || true
    if [ ! -f output/output.bin ]; then
        echo "$label | RUN FAILED (no output)" | tee -a "../../$RESULTS_FILE"
        cd ../..
        return
    fi

    # Verify
    VERIFY=$(python3 ../scripts/verify_result.py output/output.bin output/golden.bin 2>&1) || true
    if ! echo "$VERIFY" | grep -qi "pass"; then
        echo "$label | INCORRECT" | tee -a "../../$RESULTS_FILE"
        cd ../..
        return
    fi

    # Perf
    MSPROF_DIR="/tmp/msprof_sweep38_$$"
    rm -rf "$MSPROF_DIR"
    mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    timeout 120 msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_DIR" ./demo > /dev/null 2>&1 || true

    OPPROF_DIR=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        DURATION=$(awk -F',' 'NR>1 && $3+0>0{print $3}' "$OPPROF_DIR/OpBasicInfo.csv" | head -1)
        echo "$label | ${DURATION} us" | tee -a "../../$RESULTS_FILE"
    else
        echo "$label | MSPROF FAILED" | tee -a "../../$RESULTS_FILE"
    fi

    rm -rf "$MSPROF_DIR"
    cd ../..
}

# Config: label, fixsplit_line, traverse, stepM, stepN
# M=1024, N=640, K=256, 2 cores
# With FIRSTN: singleCoreM=1024, singleCoreN=320
# With FIRSTM: singleCoreM=512, singleCoreN=640
# fixSplit(sM, sN) must divide singleCoreM and singleCoreN

# Already confirmed: fix(256,128)+FIRSTN+sM2+sN1 = 74.2 us (from partial sweep above)
# Already confirmed: fix(256,64)+FIRSTN+sM2+sN1 = 88.1 us (regression)

# FIRSTN configs: singleCoreM=1024, singleCoreN=320
# Valid M tiles: 64,128,256,512,1024  Valid N tiles: 32,64,128,160,320
run_config "fix(128,128)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(128, 128, -1);" "FIRSTN" "2" "1"
run_config "fix(512,128)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(512, 128, -1);" "FIRSTN" "2" "1"
run_config "fix(256,160)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(256, 160, -1);" "FIRSTN" "2" "1"
run_config "fix(128,160)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(128, 160, -1);" "FIRSTN" "2" "1"
run_config "fix(256,320)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(256, 320, -1);" "FIRSTN" "2" "1"
run_config "fix(512,320)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(512, 320, -1);" "FIRSTN" "2" "1"
run_config "fix(1024,128)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(1024, 128, -1);" "FIRSTN" "2" "1"
run_config "fix(1024,320)+FIRSTN+sM2+sN1" "tilingApi.SetFixSplit(1024, 320, -1);" "FIRSTN" "2" "1"

# Auto tiling combos
run_config "auto+FIRSTN+sM2+sN1" "// auto tiling" "FIRSTN" "2" "1"
run_config "auto+FIRSTN+sM1+sN1" "// auto tiling" "FIRSTN" "1" "1"
run_config "auto+FIRSTM+sM2+sN1" "// auto tiling" "FIRSTM" "2" "1"
run_config "auto+FIRSTM+sM1+sN1" "// auto tiling" "FIRSTM" "1" "1"

# FIRSTM configs: singleCoreM=512, singleCoreN=640
# Valid M tiles: 64,128,256,512  Valid N tiles: 32,64,128,160,320,640
run_config "fix(256,128)+FIRSTM+sM2+sN1" "tilingApi.SetFixSplit(256, 128, -1);" "FIRSTM" "2" "1"
run_config "fix(128,128)+FIRSTM+sM2+sN1" "tilingApi.SetFixSplit(128, 128, -1);" "FIRSTM" "2" "1"
run_config "fix(256,128)+FIRSTM+sM3+sN1" "tilingApi.SetFixSplit(256, 128, -1);" "FIRSTM" "3" "1"
run_config "fix(256,320)+FIRSTM+sM2+sN1" "tilingApi.SetFixSplit(256, 320, -1);" "FIRSTM" "2" "1"
run_config "fix(256,640)+FIRSTM+sM2+sN1" "tilingApi.SetFixSplit(256, 640, -1);" "FIRSTM" "2" "1"
run_config "fix(512,128)+FIRSTM+sM2+sN1" "tilingApi.SetFixSplit(512, 128, -1);" "FIRSTM" "2" "1"
run_config "fix(512,320)+FIRSTM+sM2+sN1" "tilingApi.SetFixSplit(512, 320, -1);" "FIRSTM" "2" "1"

# stepM/stepN variations with best fixSplit
run_config "fix(256,128)+FIRSTN+sM1+sN1" "tilingApi.SetFixSplit(256, 128, -1);" "FIRSTN" "1" "1"
run_config "fix(256,128)+FIRSTN+sM3+sN1" "tilingApi.SetFixSplit(256, 128, -1);" "FIRSTN" "3" "1"
run_config "fix(256,128)+FIRSTN+sM4+sN1" "tilingApi.SetFixSplit(256, 128, -1);" "FIRSTN" "4" "1"
run_config "fix(256,128)+FIRSTN+sM2+sN2" "tilingApi.SetFixSplit(256, 128, -1);" "FIRSTN" "2" "2"
run_config "auto+FIRSTN+sM2+sN2" "// auto tiling" "FIRSTN" "2" "2"

# Restore original
cp "$BACKUP" "$ASC_FILE"

echo ""
echo "=== Sweep Complete ==="
echo ""
cat "$RESULTS_FILE"
