#!/usr/bin/env python3
"""
Iter 31 sweep: Test various kernel configurations.
Try __aicore__ with numBlocks=2 (2 physical cores), and different
tiling configs for the __mix__ mode to try to beat 110 us.
"""

import subprocess
import os
import sys
import re
import tempfile
import shutil

SOLUTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'solution')
ASC_FILE = os.path.join(SOLUTION_DIR, 'matmul_leakyrelu.asc')
BUILD_DIR = os.path.join(SOLUTION_DIR, 'build')
BEST_FILE = os.path.join(SOLUTION_DIR, 'matmul_leakyrelu.asc.best')

# Save original
with open(BEST_FILE, 'r') as f:
    ORIGINAL_CONTENT = f.read()

def run_cmd(cmd, cwd=None, timeout=120):
    """Run a shell command and return (returncode, stdout+stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, 'CMAKE_PREFIX_PATH': os.environ.get('ASCEND_HOME_PATH', '') + '/x86_64-linux/tikcpp/ascendc_kernel_cmake:' + os.environ.get('CMAKE_PREFIX_PATH', '')}
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"

def generate_kernel(mode, num_aiv, used_cores, num_blocks, fix_m, fix_n,
                    step_m, step_n, traverse, l1_cache_ub, chunk_size):
    """Generate kernel source with given configuration."""

    if mode == '__mix__':
        kernel_attr = f'__mix__(1, {num_aiv})'
        leaky_relu_block_split = f"""
    uint32_t aivBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t aivBlockNum = {num_aiv};
    uint32_t elementsPerBlock = totalElements / aivBlockNum;
    uint32_t startElem = aivBlockIdx * elementsPerBlock;
    uint32_t endElem = (aivBlockIdx == aivBlockNum - 1) ? totalElements : startElem + elementsPerBlock;
    uint32_t myElements = endElem - startElem;"""
    else:
        kernel_attr = '__aicore__'
        leaky_relu_block_split = f"""
    uint32_t startElem = 0;
    uint32_t myElements = totalElements;"""

    traverse_str = 'FIRSTN' if traverse == 'FIRSTN' else 'FIRSTM'
    l1_config = 'tilingApi.SetMatmulConfigParams(1, true);' if l1_cache_ub else ''
    fix_split = f'tilingApi.SetFixSplit({fix_m}, {fix_n}, -1);' if fix_m > 0 else ''

    return f'''/**
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
{{
    return (a + b - 1) / b;
}}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulKernel {{
public:
    __aicore__ inline MatmulKernel(){{}};
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
}};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                          GM_ADDR c, GM_ADDR workspace,
                                                                          const TCubeTiling &tiling, AscendC::TPipe *pipe)
{{
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
}}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    while (matmulObj.template Iterate<true>()) {{
        matmulObj.template GetTensorC<true>(cGlobal);
    }}
    matmulObj.End();
}}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                         int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                         int32_t &offsetBias)
{{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;
    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}}

__global__ {kernel_attr} void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, AscendC::tiling::TCubeTiling tiling)
{{
    AscendC::TPipe pipe;
    MatmulKernel<half, half, float, float> matmulKernel;
    matmulKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &matmulKernel.tiling);
    matmulKernel.Process(&pipe);

    uint32_t totalElements = tiling.M * tiling.N;
    {leaky_relu_block_split}

    const uint32_t CHUNK_SIZE = {chunk_size};
    AscendC::GlobalTensor<float> cGm;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c) + startElem, myElements);

    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(inQueue, 2, CHUNK_SIZE * sizeof(float));
    pipe.InitBuffer(outQueue, 2, CHUNK_SIZE * sizeof(float));

    uint32_t numChunks = (myElements + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (uint32_t i = 0; i < numChunks; i++) {{
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
    }}
}}

AscendC::tiling::TCubeTiling GenerateTiling(platform_ascendc::PlatformAscendC* ascendcPlatform)
{{
    using TPosition = matmul_tiling::TPosition;
    using CubeFormat = matmul_tiling::CubeFormat;
    using DataType = matmul_tiling::DataType;
    int M = 1024;
    int N = 640;
    int K = 256;
    int usedCoreNum = {used_cores};

    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    {l1_config}
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(true);
    tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::{traverse_str});
    {fix_split}
    tilingApi.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling tilingData;
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {{
        std::cout << "gen tiling failed" << std::endl;
    }}
    tilingData.stepM = {step_m};
    tilingData.stepN = {step_n};
    return tilingData;
}}

int32_t main(int32_t argc, char *argv[])
{{
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
    uint32_t numBlocks = {num_blocks};

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
}}
'''

def build_and_test(config_name):
    """Build, test correctness, and profile a configuration."""
    # Build
    rc, output = run_cmd('cmake .. && make -j4', cwd=BUILD_DIR, timeout=120)
    if rc != 0:
        return None, f"BUILD FAILED: {output[-200:]}"

    # Generate test data
    rc, output = run_cmd('python3 ../scripts/gen_data.py', cwd=BUILD_DIR)
    if rc != 0:
        return None, "GEN_DATA FAILED"

    # Run
    rc, output = run_cmd('rm -f output/output.bin && ./demo', cwd=BUILD_DIR, timeout=30)
    if rc != 0:
        return None, f"RUN FAILED: {output[-200:]}"

    # Verify correctness
    rc, output = run_cmd('python3 ../scripts/verify_result.py output/output.bin output/golden.bin', cwd=BUILD_DIR)
    if rc != 0 or 'test pass' not in output:
        return None, "INCORRECT"

    # Profile
    msprof_dir = tempfile.mkdtemp(prefix='msprof_sweep_')
    os.chmod(msprof_dir, 0o700)
    rc, output = run_cmd(
        f'msprof op --warm-up=10 --launch-count=5 --output="{msprof_dir}" ./demo',
        cwd=BUILD_DIR, timeout=120
    )

    # Parse result
    runtime = None
    try:
        for root, dirs, files in os.walk(msprof_dir):
            for f in files:
                if f.startswith('OpBasicInfo') and f.endswith('.csv'):
                    filepath = os.path.join(root, f)
                    with open(filepath, 'r') as fh:
                        for line in fh:
                            parts = line.strip().split(',')
                            if len(parts) > 2 and parts[0] == 'matmul_leakyrelu_custom':
                                try:
                                    runtime = float(parts[2])
                                except:
                                    pass
    except:
        pass

    # Cleanup
    shutil.rmtree(msprof_dir, ignore_errors=True)

    if runtime is None:
        return None, "PROFILE FAILED"

    return runtime, "OK"

# Configs to test
configs = [
    # (mode, num_aiv, used_cores, num_blocks, fix_m, fix_n, step_m, step_n, traverse, l1_cache, chunk)
    # Current best for reference
    ("__mix__", 2, 2, 1, 256, 128, 2, 1, "FIRSTN", True, 12288),

    # Try numBlocks=2 with __mix__(1,2) and usedCoreNum=4
    ("__mix__", 2, 4, 2, 256, 128, 2, 1, "FIRSTN", True, 12288),

    # Try __mix__(1,3) with 3 AIV
    ("__mix__", 3, 2, 1, 256, 128, 2, 1, "FIRSTN", True, 8192),

    # Try __mix__(1,1) with 1 AIV to reduce sync overhead
    ("__mix__", 1, 2, 1, 256, 128, 2, 1, "FIRSTN", True, 12288),

    # Try larger fix_m to reduce tile count
    ("__mix__", 2, 2, 1, 512, 128, 2, 1, "FIRSTN", True, 12288),

    # Try different chunk sizes
    ("__mix__", 2, 2, 1, 256, 128, 2, 1, "FIRSTN", True, 16384),
    ("__mix__", 2, 2, 1, 256, 128, 2, 1, "FIRSTN", True, 24576),
    ("__mix__", 2, 2, 1, 256, 128, 2, 1, "FIRSTN", True, 8192),

    # Try with NZ output format
    # ("__mix__", 2, 2, 1, 256, 128, 2, 1, "FIRSTN", True, 12288),  # NZ - need different code

    # Try step combinations
    ("__mix__", 2, 2, 1, 256, 128, 1, 2, "FIRSTN", True, 12288),
    ("__mix__", 2, 2, 1, 256, 128, 2, 2, "FIRSTN", True, 12288),
    ("__mix__", 2, 2, 1, 256, 128, 3, 1, "FIRSTN", True, 12288),
    ("__mix__", 2, 2, 1, 256, 128, 1, 1, "FIRSTN", True, 12288),

    # Try different tile sizes
    ("__mix__", 2, 2, 1, 256, 256, 2, 1, "FIRSTN", True, 12288),
    ("__mix__", 2, 2, 1, 128, 128, 2, 1, "FIRSTN", True, 12288),
    ("__mix__", 2, 2, 1, 384, 128, 2, 1, "FIRSTN", True, 12288),
    ("__mix__", 2, 2, 1, 256, 64, 2, 1, "FIRSTN", True, 12288),
    ("__mix__", 2, 2, 1, 192, 128, 2, 1, "FIRSTN", True, 12288),

    # Auto tiling (no fix split)
    ("__mix__", 2, 2, 1, 0, 0, 2, 1, "FIRSTN", True, 12288),

    # Try FIRSTM with current best
    ("__mix__", 2, 2, 1, 256, 128, 2, 1, "FIRSTM", True, 12288),

    # Try without L1 cache
    ("__mix__", 2, 2, 1, 256, 128, 2, 1, "FIRSTN", False, 12288),

    # Try larger used_cores
    ("__mix__", 2, 4, 1, 256, 128, 2, 1, "FIRSTN", True, 12288),
    ("__mix__", 2, 8, 1, 256, 128, 2, 1, "FIRSTN", True, 12288),
    ("__mix__", 2, 3, 1, 256, 128, 2, 1, "FIRSTN", True, 12288),
]

print(f"Testing {len(configs)} configurations...")
print(f"{'#':>3} {'Mode':>10} {'AIV':>3} {'Cores':>5} {'Blks':>4} {'FixM':>4} {'FixN':>4} {'StM':>3} {'StN':>3} {'Trav':>6} {'L1':>3} {'Chunk':>6} {'Time(us)':>10} {'Status':>15}")
print("-" * 120)

best_time = 110.0
best_config = None
results = []

for i, (mode, num_aiv, used_cores, num_blocks, fix_m, fix_n, step_m, step_n, traverse, l1, chunk) in enumerate(configs):
    config_name = f"cfg_{i}"

    # Generate kernel
    try:
        kernel_src = generate_kernel(mode, num_aiv, used_cores, num_blocks, fix_m, fix_n,
                                     step_m, step_n, traverse, l1, chunk)
    except Exception as e:
        print(f"{i:>3} ... GENERATE FAILED: {e}")
        results.append((i, None, str(e)))
        continue

    with open(ASC_FILE, 'w') as f:
        f.write(kernel_src)

    runtime, status = build_and_test(config_name)

    l1_str = "Y" if l1 else "N"
    time_str = f"{runtime:.1f}" if runtime else "N/A"
    print(f"{i:>3} {mode:>10} {num_aiv:>3} {used_cores:>5} {num_blocks:>4} {fix_m:>4} {fix_n:>4} {step_m:>3} {step_n:>3} {traverse:>6} {l1_str:>3} {chunk:>6} {time_str:>10} {status:>15}")

    results.append((i, runtime, status))

    if runtime and runtime < best_time:
        best_time = runtime
        best_config = i
        # Save best kernel
        with open(ASC_FILE, 'r') as f:
            best_kernel = f.read()
        with open(os.path.join(SOLUTION_DIR, 'matmul_leakyrelu.asc.sweep_best'), 'w') as f:
            f.write(best_kernel)

# Restore original
with open(ASC_FILE, 'w') as f:
    f.write(ORIGINAL_CONTENT)

print("\n" + "=" * 80)
print(f"Best: config #{best_config}, time={best_time:.1f} us")
if best_config is not None and best_time < 110.0:
    print(f"  => IMPROVEMENT FOUND! {110.0/best_time:.2f}x vs current best")
    print(f"  Config: {configs[best_config]}")
else:
    print("  => No improvement found.")

# Print summary
print("\nAll results:")
for i, runtime, status in results:
    if runtime:
        speedup = 227.9 / runtime
        print(f"  #{i}: {runtime:.1f} us ({speedup:.2f}x) - {status}")
    else:
        print(f"  #{i}: FAILED - {status}")
