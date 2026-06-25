# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
import torch_npu
has_initialized = False
from torch_npu._inductor import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_hadoop-scale-llm/xk/cxkqyn5fn4mljaj5rqqvsapacwnvdstygdqh46h47aajom7slwub.py
# Topologically Sorted Source Nodes: [mul, gw], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   gw => sum_1
#   mul => mul
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 1, 4096, 4096][16777216, 16777216, 4096, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 64, 4096, 4096][1073741824, 16777216, 4096, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %mul : Tensor "f32[1, 64, 4096, 4096][1073741824, 16777216, 4096, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sum_1 : Tensor "f32[1, 64, 4096, 1][262144, 4096, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1], True), kwargs = {})
#   return %sum_1
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_unk_fused_mul_sum_0 = async_compile.triton('triton_unk_fused_mul_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.reduction_npu_index(
    size_hints=[64, 4096, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2C', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_sum_0', 'mutated_arg_names': [], 'backend_hash': 'a15a2345395da31c737317a0b70dff821d51ec7b8ebc7f1b94460d85e1d8c1f6', 'split_axis': [0], 'tiling_axis': [1, 2], 'axis_names': ['y0', 'x1', 'r2'], 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False}
)
@triton.jit
def triton_unk_fused_mul_sum_0(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK : tl.constexpr, X1BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for y0 in range(y0_offset, min(y0_offset + Y0BLOCK, y0_numel)):
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[:,None]
            x1_mask = x1 < x1_numel
            _tmp4 = tl.full([X1BLOCK_SUB, R2BLOCK_SUB], 0, tl.float32)
            for loop_r2 in range(loops_r2):
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:]
                r2_mask = r2 < r2_numel
                tmp0 = tl.load(in_ptr0 + (r2 + 4096*x1), r2_mask & x1_mask, other=0.0)
                tmp1 = tl.load(in_ptr1 + (r2 + 4096*x1 + 16777216*y0), r2_mask & x1_mask, other=0.0)
                tmp2 = tmp0 * tmp1
                tmp3 = tl.reshape(tmp2, [X1BLOCK_SUB, R2BLOCK_SUB])
                tmp5 = _tmp4 + tmp3
                _tmp4 = tl.where(r2_mask & x1_mask, tmp5, _tmp4)
            tmp4 = tl.sum(_tmp4, 1).reshape(X1BLOCK_SUB, 1)
            tl.store(out_ptr0 + (x1 + 4096*y0 ), tmp4, x1_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_hadoop-scale-llm/hu/chuyfmwgrxt3pp2uye2ql72u4utzxlwj5qfsuilp24jz4r62m534.py
# Topologically Sorted Source Nodes: [tensor, gi_masked, gs], Original ATen: [aten._to_copy, aten.where, aten.mul]
# Source node to ATen node mapping:
#   gi_masked => where
#   gs => mul_1
#   tensor => full_default
# Graph fragment:
#   %arg2_1 : Tensor "b8[1, 64, 4096, 4096][1073741824, 16777216, 4096, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "f32[1, 1, 4096, 4096][16777216, 16777216, 4096, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[1, 64, 4096, 1][262144, 4096, 1, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %full_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[1, 64, 4096, 4096][1073741824, 16777216, 4096, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg2_1, %arg0_1, %full_default), kwargs = {})
#   %mul_1 : Tensor "f32[1, 64, 4096, 4096][1073741824, 16777216, 4096, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %arg3_1), kwargs = {})
#   return %mul_1
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_unk_fused__to_copy_mul_where_1 = async_compile.triton('triton_unk_fused__to_copy_mul_where_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[64, 4096, 4096], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2C', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_mul_where_1', 'mutated_arg_names': [], 'backend_hash': 'a15a2345395da31c737317a0b70dff821d51ec7b8ebc7f1b94460d85e1d8c1f6', 'split_axis': [0], 'tiling_axis': [1, 2], 'axis_names': ['z0', 'y1', 'x2'], 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_mul_where_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (y1_numel + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for z0 in range(z0_offset, min(z0_offset + Z0BLOCK, z0_numel)):
        for loop_y1 in range(loops_y1):
            y1 = (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None]
            y1_mask = y1 < y1_numel
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,:]
                x2_mask = x2 < x2_numel
                tmp0 = tl.load(in_ptr0 + (x2 + 4096*y1 + 16777216*z0), x2_mask & y1_mask).to(tl.int1)
                tmp1 = tl.load(in_ptr1 + (x2 + 4096*y1), x2_mask & y1_mask)
                tmp4 = tl.load(in_ptr2 + (y1 + 4096*z0), y1_mask)
                tmp2 = 0.0
                tmp3 = tl.where(tmp0, tmp1, tmp2)
                tmp5 = tmp3 * tmp4
                tl.store(out_ptr0 + (x2 + 4096*y1 + 16777216*z0), tmp5, x2_mask & y1_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        buf0 = empty_strided((1, 64, 4096, 1), (262144, 4096, 1, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [mul, gw], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_unk_fused_mul_sum_0.run(arg0_1, arg1_1, buf0, 64, 4096, 4096, stream=stream0)

        buf1 = empty_strided((1, 64, 4096, 4096), (1073741824, 16777216, 4096, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [tensor, gi_masked, gs], Original ATen: [aten._to_copy, aten.where, aten.mul]
        stream0 = get_raw_stream(0)
        triton_unk_fused__to_copy_mul_where_1.run(arg2_1, arg0_1, arg3_1, buf1, 64, 4096, 4096, stream=stream0)



        return (buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 4096, 4096), (16777216, 16777216, 4096, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 64, 4096, 4096), (1073741824, 16777216, 4096, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 64, 4096, 4096), (1073741824, 16777216, 4096, 1), device='npu:0', dtype=torch.bool)
    arg3_1 = rand_strided((1, 64, 4096, 1), (262144, 4096, 1, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
