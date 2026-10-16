# 2026-10-16 NVIDIA CUDA 故障诊断与恢复边界

## 实测状态

- 当前内核 `6.8.0-147-generic`；Ubuntu 24.04；RTX 3050 Mobile 在 PCI 总线上可见。
- `nvidia-smi` 无法连接驱动；项目环境 PyTorch `2.10.0+cu130` 的 `torch.cuda.is_available()` 为 `False`，设备数为 0。因此没有进行 CUDA 张量运算或恢复正式训练。
- 系统通过 APT 安装 `nvidia-driver-595-open` / `nvidia-kernel-source-595-open`，版本 `595.91.07`。没有安装 `dkms` 命令，不是可直接触发的 DKMS 安装。当前内核的 headers 已安装，但其 `/lib/modules/6.8.0-147-generic/` 下没有 `nvidia.ko`；`modinfo nvidia` 失败，`lsmod` 中也没有 `nvidia` 或 `nouveau`。因此当前首要故障是**缺少与运行内核匹配的 NVIDIA 模块**，并非已经加载的模块被占用。
- Secure Boot 已启用。启动日志没有出现 NVIDIA 模块签名被拒绝；目前根本没有可加载的 147 模块，不能把故障归因为签名。旧内核 `6.8.0-146-generic` 的 NVIDIA 595.91.07 模块存在，`modinfo -k` 显示由 Canonical 签名；该内核映像与 initrd 也存在。
- APT 有 `linux-modules-nvidia-595-open-6.8.0-147-generic` 候选包，但它是针对 `595.99.02` 构建，依赖对应的 `nvidia-kernel-common-595`。安装模拟将升级 16 个同分支驱动/用户态包，并移除旧 142 模块；这不是只补一个当前内核模块的零风险操作。没有执行安装、卸载、GRUB 修改或重启。
- 当前用户属于 sudo 组，但 `sudo -n` 返回需要密码；本次会话没有执行系统包安装所需的管理员授权。根分区剩余约 15 GB，项目所在 `/home` 剩余约 29 GB；磁盘空间不是本次故障原因。

## 训练与数据保护

死亡机制研究已完成并推送。课程数据集有 210 个快照，SHA256 `3d6639b237f1df44b344e6b3fd5f6d8ea325d9d9164d7bed57dd77562a896bac`，文件哈希复核通过。Normal-Start 与 Curriculum 各有 1 个已提交的 4096 样本任务；两份任务 manifest、模型、优化器、RNG 和 8-worker 状态均通过 `verify_task`。冻结 PPO30M SHA256 也通过。正式 500K 训练尚未开始。

后台 `tetris-failure-curriculum` 会话在 `waiting_for_gpu_driver`，只做只读 GPU 检查；不会退化为 CPU 训练，也不会重新生成数据或清空事务。若机器重启，tmux 会话本身不会跨重启保留，需重新启动原队列；已提交任务可恢复。

## 最小风险恢复方案

本会话无法在不取得管理员权限的情况下，为运行中的 147 内核安装匹配的预编译模块。**当前不作系统修改。** 管理员可择一执行：

1. 在维护窗口人工选择已安装的 `6.8.0-146-generic` 启动（旧内核及 Canonical 签名的 595.91.07 模块均在），不修改 GRUB 默认项；这需要一次人工重启。
2. 如必须继续用 147 内核，由管理员审核 APT 事务后安装与 147 匹配的 Ubuntu 预编译 `595-open` 模块和同分支一致的用户态组件。此方案可能升级多个驱动包，执行前应确认维护窗口和回退路径。不可直接复制旧 146 的 `.ko` 到 147。

恢复后先逐项通过 `nvidia-smi`、PyTorch CUDA 可用性、RTX 3050 识别与 GPU 张量加法，再从两个 4096 样本已提交事务继续原训练队列。没有通过验证前，不宣布训练已经恢复。

证据：`runs/ppo30m_failure_curriculum/status.json`、`normal/committed/task_000001/`、`curriculum/committed/task_000001/`、`curriculum_manifest.json`；本次终端命令包含 `uname`、`modinfo`、`mokutil`、`dpkg`、`apt-cache`、`apt-get -s`、`journalctl`、`nvidia-smi`、PyTorch CUDA 检查。
