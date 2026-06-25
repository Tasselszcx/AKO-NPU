# Cases 复制规则

从运行实例复制到 cases/ 时使用以下命令：

```bash
rsync -a \
  --exclude='.git' \
  --exclude='skills/' \
  --exclude='**/build/' \
  --exclude='**/*.o' \
  --exclude='**/*.bin' \
  --exclude='**/*.so' \
  --exclude='**/*.a' \
  --exclude='**/OPPROF_*' \
  --exclude='**/PROF_*' \
  --exclude='**/msprof_*' \
  /path/to/AKO-NPU-run-xxx/ cases/AKO-NPU-run-xxx/
```

## 保留的内容
- `solution/` — 最终优化后的算子源码
- `scripts/` — bench.sh 等评测脚本
- `trajectory/` — 每轮迭代的代码快照 + output.txt
- `input/` — 原始输入
- `ITERATIONS.md` — 迭代日志
- `.claude/settings.local.json` — 权限配置

## 排除的内容（编译产物/大文件）
- `.git/` — 避免嵌套 git
- `skills/` — CANN Skills 仓库（太大，不复制）
- `**/build/` — 编译产物目录
- `**/*.o` `**/*.bin` `**/*.so` `**/*.a` — 二进制文件
- `**/OPPROF_*` `**/PROF_*` `**/msprof_*` — msprof 原始输出
