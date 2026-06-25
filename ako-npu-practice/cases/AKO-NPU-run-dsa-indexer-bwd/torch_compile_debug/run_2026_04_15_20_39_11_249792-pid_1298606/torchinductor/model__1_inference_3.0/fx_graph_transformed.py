class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 1, 4096, 4096]", arg1_1: "f32[1, 64, 4096, 4096]", arg2_1: "b8[1, 64, 4096, 4096]", arg3_1: "f32[1, 64, 4096, 1]"):
         # 
        mul: "f32[1, 64, 4096, 4096]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg1_1 = None
        sum_1: "f32[1, 64, 4096, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True);  mul = None
        
         # 
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[1, 64, 4096, 4096]" = torch.ops.aten.where.self(arg2_1, arg0_1, full_default);  arg2_1 = arg0_1 = full_default = None
        
         # 
        mul_1: "f32[1, 64, 4096, 4096]" = torch.ops.aten.mul.Tensor(where, arg3_1);  where = arg3_1 = None
        return (sum_1, mul_1)
        