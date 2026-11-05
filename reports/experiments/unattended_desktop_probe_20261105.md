# 轻薄本无人值守：桌面通信暂停实测

用户首要任务为配置轻薄本无人值守。此前目标paused造成定时器返回goal_paused；按用户本次指令一次性恢复active，配置后台/每小时续办/5分钟自检及开机驻留、防休眠，并补强健康检查。29项回归通过，当前自动续办于2026-11-05 22:48:17执行了真实本机命令。

## 实测

- 22:53:52，独立systemd测试服务将唯一桌面代理PID13928暂停。限定同用户、准确命令、PID/starttime和pidfd，后台PID13827不属于目标。
- 22:53:56，AI读取测试状态后实际执行新的工具命令。代理为T；工具进程祖先187119→187118→13827→1，证实执行来自原生后台。
- 22:54:22，AI再次通过本地Unix接口读取目标active，代理仍为T。两次工具调用相隔26秒，非预先写好的单个shell任务。
- 同秒达到30秒时限，代理自动恢复。随后13928为S，13827的PID/starttime及父进程1均未变；测试服务MainPID0/Resultsuccess。

恢复由Python finally与systemd ExecStopPost托管，服务40秒硬上限；CPU10%/128MiB并入总计算slice，无训练。18项代理/后台定向测试通过，覆盖PID复用拒绝、部分暂停失败恢复、重复恢复、代理重连/未暂停/非后台父链拒绝等。测试服务只手动运行一次，未enable，不在开机时再次暂停桌面。

## 证据和限制

`runs/unattended_desktop_probe_20261105/design.json`、`observation.json`、`followup_observation.json`、`state.json`及`summary.json`；原配置验收为`runs/autonomous_development_20261105/unattended_configuration_receipt.json`。

已验证桌面代理通信暂停时，原生Codex后台仍能继续AI工具执行，且桌面代理恢复正常。该测试没有关闭游戏本、断开物理网络或重启轻薄本，不能声称这些完整场景已实测。用户未来暂停/取消仍应停止；此次恢复不是覆盖未来暂停的授权。下一步保持既有后台机制并继续已授权开发，实际断开后如获得新证据再记录，避免重复本测试。
