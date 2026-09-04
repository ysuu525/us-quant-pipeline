"""``evaluate_fold.py --gpu-mem-fraction`` 与 ``runtime`` 键的单元测试。

**全程 CPU-only**：不建任何 CUDA 张量、不调 ``set_per_process_memory_fraction``
的真身（一律 monkeypatch 成记录调用的桩）、不读显存统计。
``torch.device("cuda")`` 只是一个描述符对象，构造它不会初始化 CUDA 上下文，
本文件用它来走「device 判定为 cuda」的分支。

盯死三件事：
1. ``--gpu-mem-fraction`` 默认 None；越界值（<=0 或 >1）由 argparse 拒绝；
2. device 为 cuda 时封顶函数被以正确的值调用一次，cpu / mps 时一次都不调；
3. ``runtime`` 键在 CPU 路径下 allocated / reserved 为 None，且键集固定
   （scoring_config 的口径核对靠的是另一个字典，runtime 只放运行侧事实）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "evaluate_fold.py"
SPEC = importlib.util.spec_from_file_location("evaluate_fold_under_test", SCRIPT)
EF = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EF
SPEC.loader.exec_module(EF)

REQUIRED = [
    "--model-dir", "m", "--processed", "p",
    "--val-start", "2022-01-03", "--val-end", "2022-06-30",
    "--lookback", "90", "--tag", "unit_test",
]

RUNTIME_KEYS = {
    "gpu_mem_fraction", "device", "scoring_seconds",
    "max_memory_allocated_gib", "max_memory_reserved_gib",
    "pytorch_cuda_alloc_conf",
}


# ---------------------------------------------------------------- argparse

def test_default_is_none():
    args = EF.build_parser().parse_args(REQUIRED)
    assert args.gpu_mem_fraction is None


@pytest.mark.parametrize("val", ["0.70", "0.1", "1", "1.0"])
def test_valid_fractions_parse(val):
    args = EF.build_parser().parse_args(REQUIRED + ["--gpu-mem-fraction", val])
    assert args.gpu_mem_fraction == pytest.approx(float(val))


@pytest.mark.parametrize("val", ["0", "0.0", "1.5", "-0.3", "2", "abc"])
def test_out_of_range_fraction_is_rejected(val):
    with pytest.raises(SystemExit):
        EF.build_parser().parse_args(REQUIRED + ["--gpu-mem-fraction", val])


def test_help_text_marks_it_engineering_only():
    """帮助里必须写明「不进 scoring_config / 不改数值」——§八 的口径核对
    只认 scoring_config，这条参数越界会直接污染口径可比性。"""
    txt = EF.build_parser().format_help()
    assert "scoring_config" in txt


# ------------------------------------------------------- apply_gpu_mem_fraction

class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, fraction, *a, **kw):
        self.calls.append(fraction)


@pytest.fixture
def spy(monkeypatch):
    s = _Spy()
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", s)
    return s


def test_cuda_device_sets_cap(spy):
    assert EF.apply_gpu_mem_fraction(0.70, torch.device("cuda")) is True
    assert spy.calls == [pytest.approx(0.70)]


def test_cuda_device_with_index_sets_cap(spy):
    assert EF.apply_gpu_mem_fraction(0.55, torch.device("cuda:0")) is True
    assert spy.calls == [pytest.approx(0.55)]


def test_cuda_as_plain_string_sets_cap(spy):
    """--device 传进来的是字符串；判定必须对字符串同样成立。"""
    assert EF.apply_gpu_mem_fraction(0.80, "cuda") is True
    assert EF.apply_gpu_mem_fraction(0.80, "cuda:1") is True
    assert spy.calls == [pytest.approx(0.80), pytest.approx(0.80)]


@pytest.mark.parametrize("device", ["cpu", "mps", None])
def test_non_cuda_device_never_calls_cap(spy, device):
    dev = device if device is None else torch.device(device)
    assert EF.apply_gpu_mem_fraction(0.70, dev) is False
    assert spy.calls == []


def test_none_fraction_never_calls_cap(spy):
    """默认（不传参）路径：即使 device 是 cuda 也一个调用都不发 → 逐位不变。"""
    assert EF.apply_gpu_mem_fraction(None, torch.device("cuda")) is False
    assert spy.calls == []


# -------------------------------------------------------- build_runtime_info

def test_runtime_on_cpu_has_no_memory_numbers(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    rt = EF.build_runtime_info(None, torch.device("cpu"), 12.5)
    assert set(rt) == RUNTIME_KEYS
    assert rt["max_memory_allocated_gib"] is None
    assert rt["max_memory_reserved_gib"] is None
    assert rt["gpu_mem_fraction"] is None
    assert rt["device"] == "cpu"
    assert rt["scoring_seconds"] == pytest.approx(12.5)
    assert rt["pytorch_cuda_alloc_conf"] is None


def test_runtime_records_fraction_and_env(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    rt = EF.build_runtime_info(0.70, torch.device("cpu"), 1.0)
    assert rt["gpu_mem_fraction"] == pytest.approx(0.70)
    assert rt["pytorch_cuda_alloc_conf"] == "garbage_collection_threshold:0.8"
    # CPU 路径仍然不读显存统计
    assert rt["max_memory_allocated_gib"] is None


def test_runtime_cpu_path_issues_no_cuda_query(monkeypatch):
    """CPU 路径一个 CUDA 调用都不能有：把 is_available 换成会炸的桩来证明。"""
    def _boom():  # pragma: no cover - 被调用即测试失败
        raise AssertionError("CPU 路径不得查询 CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    rt = EF.build_runtime_info(None, torch.device("cpu"), 0.5)
    assert rt["max_memory_allocated_gib"] is None
    assert EF.apply_gpu_mem_fraction(0.7, torch.device("cpu")) is False


def test_runtime_cuda_path_reads_peak_stats(monkeypatch):
    """device=cuda 的读数路径：桩掉三个 CUDA 查询，验证换算为 GiB。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 5 << 30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 7 << 30)
    rt = EF.build_runtime_info(0.70, torch.device("cuda"), 2.0)
    assert rt["max_memory_allocated_gib"] == pytest.approx(5.0)
    assert rt["max_memory_reserved_gib"] == pytest.approx(7.0)
    assert rt["device"] == "cuda"


# ------------------------------------------------------------------ 口径纪律

def test_scoring_config_keys_are_untouched():
    """§八：metrics.json 的 scoring_config 是机器核对口径的字段，
    本次改动不得往里加键。源码级断言，避免今后误把 runtime 并进去。"""
    txt = SCRIPT.read_text(encoding="utf-8")
    head = txt.split('"scoring_config": {', 1)[1].split("},", 1)[0]
    for key in ("amp", "batch_size", "sample_count", "lookback", "predict"):
        assert f'"{key}"' in head
    assert "gpu_mem_fraction" not in head
    assert "runtime" not in head
