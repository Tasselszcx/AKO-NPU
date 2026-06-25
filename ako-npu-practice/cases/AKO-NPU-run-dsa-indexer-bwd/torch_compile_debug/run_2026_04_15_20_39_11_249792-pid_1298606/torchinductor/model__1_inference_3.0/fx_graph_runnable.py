
import os
os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True,base_addr_aligned_kb:0'
os.environ['TORCH_NPU_ALLOW_INTERNAL_FORMAT'] = '0'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_hadoop-scale-llm'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.0
# torch cuda version: None
# torch git version: f4a3e9a5ad5386e1e35c3b516fa63032a974ca03


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor(0.))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg1_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul, [-1], True);  mul = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(arg2_1, arg0_1, full_default);  arg2_1 = arg0_1 = full_default = None
        mul_1 = torch.ops.aten.mul.Tensor(where, arg3_1);  where = arg3_1 = None
        return (sum_1, mul_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 67108864, device=device(type='npu', index=0))
    reader.tensor(buf0, (1, 1, 4096, 4096), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4294967296, device=device(type='npu', index=0))
    reader.tensor(buf1, (1, 64, 4096, 4096), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1073741824, device=device(type='npu', index=0), dtype_hint=torch.bool)
    reader.tensor(buf2, (1, 64, 4096, 4096), dtype=torch.bool, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1048576, device=device(type='npu', index=0))
    reader.tensor(buf3, (1, 64, 4096, 1), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)