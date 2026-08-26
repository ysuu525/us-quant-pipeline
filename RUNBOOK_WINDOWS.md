# Windows（4080 Super）RUNBOOK

从零到开训的完整步骤。原则：**这台机器只负责跑，不负责调**——任何测试
不绿、冒烟不过，都回 Mac 修好再拉。

## 0. 前置（一次性）

- NVIDIA 驱动为最新 Game Ready / Studio 版（`nvidia-smi` 能看到 4080 Super）；
- Git for Windows；Python 3.10–3.12（python.org 安装时勾选 Add to PATH）；
- 建议：把仓库放本地盘（不要放 OneDrive 同步目录），并把仓库目录加进
  Windows Defender 排除项（否则每个 checkpoint 写盘都被扫描，训练变慢）。

## 1. 克隆 + 环境（一条命令）

```powershell
git clone --recurse-submodules https://github.com/ysuu525/us-quant-pipeline.git
cd us-quant-pipeline
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

脚本做完：venv、CUDA torch、依赖、submodule、CUDA 自检、`pytest` 全绿、
`--smoke` 冒烟。任何一步失败即中止并给出原因。

之后所有命令用 venv 的 Python：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. 预训练权重（可选预下载）

首次训练会自动从 HuggingFace 拉取。想提前下好（或机器不便联网时拷入）：

```powershell
python -c "from huggingface_hub import snapshot_download as d; d('NeoQuasar/Kronos-Tokenizer-base'); d('NeoQuasar/Kronos-small')"
```

缓存在 `%USERPROFILE%\.cache\huggingface`。离线机器：在有网机器下好后
拷整个缓存目录，或把模型目录路径直接传 `--pretrained-tokenizer/--pretrained-predictor`。

## 3. 数据就位后：配置指路

CRSP 快照目录（`crsp_ciz_YYYY-MM-DD_...`，含 `metadata/snapshot_manifest.json`
的完整快照）放到本地盘后，新建 `configs\local.yaml`（不进 git）：

```yaml
paths:
  snapshot_dir: "D:/data/crsp_ciz_2026-XX-XX_..."
snapshot:
  snapshot_id: "对应 manifest 里的快照 ID"
```

> 训练入口吃的是整理后的面板 parquet（未过滤全量 + 交易日历表）。
> 从快照原始层到面板的准备脚本随 Phase 5（数据核对阶段）一起进仓库，
> 届时本节补充一条命令。

## 4. 训练（每折 × lookback × seed 一条命令）

```powershell
python -m kronos_ft.train `
  --panel <面板.parquet> --index-parquet <市场指数.parquet> `
  --train-start 2015-01-02 --train-end 2017-12-22 `
  --lookback 90 --seed 0 --stage both `
  --out outputs\fold01_lb90_s0
```

- 消融顺序按 `docs/预注册_v1.md` §2：先粗筛折 × lookback {60,90,200,400} ×
  seeds {0,1,2}，存活档再全折；
- 每次运行自动追加一行到 `experiments/ledger.md`（登记簿，append-only，
  随 git 提交）；
- 产物在 `--out`：逐 epoch checkpoint、`*_final/`（SWA 权重平均后的最终
  模型）、`*_summary.json`（早停/最优 epoch/内层 loss 曲线）。

## 5. 推理打分（训练后）

推理走 `kronos_ft.infer.run_scoring`（多路径均值，官方采样参数），按折
在验证窗上出全 universe 逐日 scores 并落盘 parquet——批量驱动脚本随
Phase 5 数据准备一起补上；执行层消融（N/触发规则/weekday）在 scores 上
用 `crsp_pipeline.execution_sim` 跑，CPU 即可，Mac/Windows 均可。

## 6. 常见问题

| 症状 | 处理 |
| --- | --- |
| `torch.cuda.is_available()` 为 False | 先 `nvidia-smi` 确认驱动；再确认装的是 cu 版 wheel：`pip show torch` 版本号应带 `+cu12x` |
| torch 安装时 404 / 找不到版本 | PyTorch 换了 CUDA 档位。去 pytorch.org 首页拿当前 Windows+CUDA 的 index-url，替换 setup 脚本第 3 步重跑 |
| pytest 有红 | 停。不要在这台机器上改逻辑，回 Mac 修好 push，这边重新 pull |
| DataLoader 卡住/报 spawn 错误 | 默认 `num_workers=0` 不会发生；调大后出问题就改回 0（吞吐瓶颈在 GPU，不在加载） |
| 训练中途断电/中断 | 逐 epoch checkpoint 都在 `--out` 里；当前版本无自动续训，重跑该配置即可（登记簿会留两条记录，正常） |
| 显存不够（lookback=400 时） | 降 `--batch-size`（32→16→8）；4080 Super 16GB 对 Kronos-small 余量充足，一般不会遇到 |
| 长时间训练 | Windows 电源计划设为高性能、关闭睡眠；不要用远程桌面断开时锁 GPU 的省电策略 |
