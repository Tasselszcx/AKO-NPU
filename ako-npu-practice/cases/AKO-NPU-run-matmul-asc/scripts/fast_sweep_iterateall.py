#!/usr/bin/env python3
"""Fast sweep of IterateAll configs — skip msprof, just run 3x and take min."""
import subprocess, os, sys, re, time

SOLUTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solution")

TEMPLATE = r"""/**
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
    matmulObj.template IterateAll<true>(cGlobal);
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

__global__ __mix__(1, 2) void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, AscendC::tiling::TCubeTiling tiling)
{{
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
    tilingApi.SetTraverse(matmul_tiling::{traverse});
{fix_split_line}
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
}}
"""

def run_config(cfg):
    fix_split = cfg.get("fix_split")
    fix_split_line = f"    tilingApi.SetFixSplit({fix_split[0]}, {fix_split[1]}, -1);" if fix_split else ""

    src = TEMPLATE.format(
        chunk_size=cfg["chunk_size"],
        traverse=cfg["traverse"],
        fix_split_line=fix_split_line,
        step_m=cfg["step_m"],
        step_n=cfg["step_n"],
    )

    asc_path = os.path.join(SOLUTION_DIR, "matmul_leakyrelu.asc")
    with open(asc_path, "w") as f:
        f.write(src)

    build_dir = os.path.join(SOLUTION_DIR, "build")
    subprocess.run(["rm", "-rf", build_dir], check=True)

    env = os.environ.copy()
    ascend_home = env.get("ASCEND_HOME_PATH", "")
    env["CMAKE_PREFIX_PATH"] = f"{ascend_home}/x86_64-linux/tikcpp/ascendc_kernel_cmake:{env.get('CMAKE_PREFIX_PATH', '')}"

    try:
        os.makedirs(build_dir, exist_ok=True)
        r = subprocess.run(["cmake", ".."], cwd=build_dir, capture_output=True, text=True, timeout=60, env=env)
        if r.returncode != 0: return None, "cmake"
        r = subprocess.run(["make", "-j4"], cwd=build_dir, capture_output=True, text=True, timeout=120, env=env)
        if r.returncode != 0: return None, "build"
    except: return None, "build_timeout"

    try:
        subprocess.run(["python3", "../scripts/gen_data.py"], cwd=build_dir, capture_output=True, timeout=30)
    except: return None, "gen_data"

    # Run and verify
    try:
        r = subprocess.run(["./demo"], cwd=build_dir, capture_output=True, text=True, timeout=30)
        if r.returncode != 0: return None, "run"
    except: return None, "run_timeout"

    try:
        r = subprocess.run(["python3", "../scripts/verify_result.py", "output/output.bin", "output/golden.bin"],
                          cwd=build_dir, capture_output=True, text=True, timeout=30)
        if "test pass" not in r.stdout: return None, "incorrect"
    except: return None, "verify"

    # Profile with msprof
    import tempfile, glob
    prof_dir = tempfile.mkdtemp(prefix="msprof_fs_")
    os.chmod(prof_dir, 0o700)
    try:
        r = subprocess.run(["msprof", "op", "--warm-up=10", "--launch-count=5", f"--output={prof_dir}", "./demo"],
                          cwd=build_dir, capture_output=True, text=True, timeout=120)
        output = r.stdout + r.stderr
        durations = re.findall(r'Task Duration\(us\):\s*([\d.]+)', output)
        if durations:
            return float(durations[-1]), "ok"
        csv_files = glob.glob(f"{prof_dir}/OPPROF_*/OpBasicInfo.csv")
        if csv_files:
            with open(csv_files[0]) as f:
                lines = f.read().strip().split('\n')
            if len(lines) > 1:
                d = re.findall(r'(\d+\.\d+)', lines[1])
                if d: return float(d[0]), "ok"
        return None, "no_duration"
    except: return None, "msprof"

configs = []

# Core sweep: tile sizes with FIRSTN
for fs in [(256,128), (256,64), (128,128), (512,64), None]:
    for sm, sn in [(2,1), (1,1), (2,2), (3,1)]:
        configs.append({"fix_split": fs, "step_m": sm, "step_n": sn,
                       "traverse": "MatrixTraverse::FIRSTN", "chunk_size": 12288})

# FIRSTM variants
for fs in [(256,128), (128,128), None]:
    for sm, sn in [(2,1), (1,1)]:
        configs.append({"fix_split": fs, "step_m": sm, "step_n": sn,
                       "traverse": "MatrixTraverse::FIRSTM", "chunk_size": 12288})

# Chunk sizes with best tiling
for cs in [8192, 10240, 14336]:
    configs.append({"fix_split": (256,128), "step_m": 2, "step_n": 1,
                   "traverse": "MatrixTraverse::FIRSTN", "chunk_size": cs})

print(f"Testing {len(configs)} configs...", flush=True)
results = []
best_rt = 999

for i, cfg in enumerate(configs):
    fs = f"{cfg['fix_split']}" if cfg['fix_split'] else "auto"
    desc = f"fix={fs} step=({cfg['step_m']},{cfg['step_n']}) {cfg['traverse'].split('::')[1]} chunk={cfg['chunk_size']}"

    rt, status = run_config(cfg)
    if rt is not None:
        marker = " ***BEST***" if rt < best_rt else ""
        if rt < best_rt: best_rt = rt
        print(f"[{i+1}/{len(configs)}] {rt:.1f} us  {desc}{marker}", flush=True)
        results.append((rt, desc, cfg))
    else:
        print(f"[{i+1}/{len(configs)}] FAIL({status})  {desc}", flush=True)

print(f"\n=== TOP 10 ===", flush=True)
results.sort()
for rt, desc, cfg in results[:10]:
    print(f"  {rt:.1f} us  {desc}", flush=True)
print(f"\nBest: {best_rt:.1f} us", flush=True)
