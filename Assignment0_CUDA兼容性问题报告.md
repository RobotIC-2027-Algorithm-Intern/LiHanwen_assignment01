# CUDA 版本建议在 RTX 50 系显卡上的问题报告（供组内讨论）

> 报告人：RobotIC 算法组 · 视觉（YOLO）教材编写
> 日期：2026-09-29
> 状态：**待组内讨论后决定处理办法**
> 相关项目：Assignment-0 预习作业、本年度视觉 YOLO 教材 2.1 节

------------------------------------------------------------------------

## 0. 结论摘要

1. **定位修正**：“**CUDA 版本超过 12.4，仍然选择 12.4**”这条建议，**只存在于 Assignment-0 的
   `Windows版教程.docx`** ，Linux 版教程中**没有**这句话。
2. **我们的问题**：本年度 YOLO 教材在编写时，**把这条 Windows 版规则机械地套用到了 Linux 部分**，
   这属于我们的误用，不是 Assignment-0 教程本身的问题。
3. **但确有一个真实的技术问题**：无论 Windows 还是 Linux，这条规则在 **RTX 50 系（Blackwell 架构，
   算力 sm_120）** 显卡上都可能导致**“装得上、却用不了”**的环境——`torch.cuda.is_available()`
   返回 `True`，但真正训练时报 `no kernel image is available`。

## 1. 问题在 Assignment-0 中的具体位置

| 文件 / 位置 | 原文（摘录） |
| --- | --- |
| `Windows版教程.docx` → **「四.配置环境（下）」** | “**值得注意的是，如果你的CUDA版本超过12.4，仍然选择12.4。**” |
| `Windows版教程.docx` → 「四.配置环境（下）」 | “输入 `nvidia-smi`，查看你的 CUDA 版本…随后访问 <https://pytorch.org/get-started/locally/>，按照自己的 CUDA 版本选择不同的选项…” |
| `Linux版教程.docx` → **「四.配置环境（下）」** | 只写“按照自己的 CUDA 版本选择不同的选项，然后复制红框里面的命令…”——**没有**“超过 12.4 仍选 12.4”这句 |
| `README.md` → 「具体要求」「注」 | “…完成 `OpenCV、jupyter Pytorch GPU版` 的安装…”“电脑没有 Nvidia GPU，或 Pytorch 版本安装不正确的同学会在 `test.py` 中显示 `CUDA is currently unavailable`…” |

> 也就是说：**这条规则是 Windows 版教程专属的**；当年的示例 CUDA 版本正好是 12.4/12.6。

## 2. 复现环境

- 操作系统：**Ubuntu 22.04**（内核 `Linux-6.8.0-87-generic`，`glibc 2.35`）
- 显卡：**NVIDIA GeForce RTX 5060 Laptop GPU**（Blackwell 架构，算力 **sm_120**，显存 8 GB）
- 驱动：`NVIDIA-SMI 580.105.08`，`CUDA Version: 13.0`
- Python：conda 虚拟环境，Python 3.10

## 3. 复现步骤与结果

按“超过 12.4 仍选 12.4”的思路选择安装命令：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

结果：

- 安装**成功**：`torch-2.6.0+cu124`、`torchvision-0.21.0+cu124`
- `torch.cuda.is_available()` → **`True`**（**具有误导性**，看起来一切正常）
- 实际执行 GPU 运算时**失败**：

```text
UserWarning: NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not compatible with
the current PyTorch installation. The current PyTorch install supports CUDA capabilities
sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
...
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

## 4. 原因分析

cu124 版 PyTorch 的二进制**只编译到 sm_90**，而 RTX 50 系是 **Blackwell（sm_120）**，
没有对应的 GPU kernel。于是形成“三层误导”：

1. `nvidia-smi` 正常 → 看起来“有 GPU”；
2. `torch.cuda.is_available()` 返回 `True` → 看起来“配好了”；
3. 只有真正训练/计算时才崩 → 新人几乎无法自行定位。

这也与 `README.md` 的预期不符：按 README，安装不正确会显示 `CUDA is currently unavailable`，
但**“版本过旧”不会显示 unavailable**，而是推迟到训练时报错。

## 5. 解决办法（已实机验证）

改用**支持 sm_120 的较新 CUDA 版本**（例如 **cu128**）：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

同一台机器实测：

- `torch-2.11.0+cu128`、`torchvision-0.26.0+cu128`
- `torch.cuda.is_available()` → `True`，且**实际 GPU 矩阵运算成功**
- `yolo checks`：`CUDA:0 (NVIDIA GeForce RTX 5060 Laptop GPU, 7681MiB)`

> 通用原则：**显卡型号与 CUDA 版本要对应**，以 <https://pytorch.org/get-started/locally/>
> 按“当前显卡/驱动”生成的命令为准，不要手工把版本“降”下来。

## 6. 本教材已采取的处理

- **Linux 部分**：不再套用“>12.4 仍选 12.4”，改为强调“**按显卡选择 CUDA 版本**”，并补一条
  RTX 50 系（Blackwell）的提醒。
- **Windows 部分**：**保留**“CUDA 版本超过 12.4 仍然选择 12.4”这段（与 Assignment-0 保持一致），
  但在教材中留了一条**待验证提醒**：请负责 Windows 测试的同学重点验证 **50 系显卡**。

## 7. 建议组内讨论的问题

1. Assignment-0 的 Windows 版教程是否也要补充 **50 系显卡需较新 CUDA** 的说明？
2. 是否统一要求新人安装后用**一次真实 GPU 运算**验收（而不是只看 `is_available()`）？
3. 是否需要更新 `test.py`，让它也能检出“版本过旧导致 GPU 不可用”？

## 附：另一处发现的坑（CPU 版与 GPU 版不能直接切换）

若环境里**已装了 CPU 版 torch**，再执行 GPU 版安装命令，pip 会认为 `torch` 已满足而
**跳过安装**。需要先卸载：

```bash
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
