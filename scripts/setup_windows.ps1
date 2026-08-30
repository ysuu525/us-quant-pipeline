# Windows (4080 Super) 一键环境搭建。在仓库根目录用 PowerShell 运行：
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# 幂等：重复运行安全。

$ErrorActionPreference = "Stop"

# 0. 基本检查
git --version | Out-Null
$py = "python"
& $py -c "import sys; assert sys.version_info >= (3,10), sys.version" 2>$null
if ($LASTEXITCODE -ne 0) { throw "需要 Python >= 3.10（python -V 检查）" }

# 长路径保险（仓库含中文文件名/深目录）
git config core.longpaths true

# 1. submodule（官方 Kronos，钉死 commit）
git submodule update --init

# 2. venv
if (-not (Test-Path ".venv")) { & $py -m venv .venv }
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet

# 3. CUDA 版 torch（4080 Super；若此 index 失效，见 RUNBOOK_WINDOWS.md 常见问题）
& $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. 本项目（含 dev/train 依赖）
& $venvPy -m pip install -e ".[dev,train]"

# 5. 验证：CUDA 可用 + 测试全绿 + 训练冒烟
& $venvPy -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用：检查 NVIDIA 驱动'; print('CUDA OK:', torch.cuda.get_device_name(0))"
& $venvPy -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest 未全绿——停下，逻辑问题回 Mac 修" }
& $venvPy -m kronos_ft.train --smoke
if ($LASTEXITCODE -ne 0) { throw "训练冒烟失败" }

Write-Host ""
Write-Host "=== 环境就绪。后续步骤见 RUNBOOK_WINDOWS.md ===" -ForegroundColor Green
