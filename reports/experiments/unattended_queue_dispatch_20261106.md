# 轻薄本空闲队列唤醒修复

2026-11-06本机核验发现：02:27普通额度恢复、原目标active并成功queue后，消息一直pending至08:57。此前后台进程与timer正常、短时断开桌面仍执行的证据，均不能覆盖这条恢复链路。

原投递器没有线程idle/notLoaded时消费队列的步骤。现保留每小时新增投递和原去重，额外在每分钟本机轻检查中通过官方RPC读取线程状态、队列与goal；仅对已授权原聊天/本机工作副本在idle时queue/start消费原消息，notLoaded先恢复原配置。active不打断，暂停/取消、STOP、非本机环境拒绝；启动前普通额度必须明确允许，失败或额度不足等3600秒。不创建新线程、不换模型/账户、不改预算或用量、不重启健康后台。

官方协议说明线程读取与恢复分别承担状态查询与重新加载，具体queue/start字段取自本机Codex0.160.1生成的实验协议Schema，未臆造API。参考[App Server官方文档](https://learn.chatgpt.com/docs/app-server)。

保护回归41项通过，覆盖idle/unloaded、忙碌不打断、额度拒绝/未知、暂停竞争、STOP竞争、错误副本、网络失败间隔及恢复竞争。原评测模板缺Install也已补齐，完成态实例enabled，真实再次启动already_completed，无重算。

09:05:17真实小时服务执行新路径，already_queued→thread_busy，尚未执行queue/start。本轮结束后由既有timer自动尝试；验收需检查hourly.jsonl中queued_start对应turn_id，并在下一实际AI工具调用追加证据。accepted不是开发执行证明；本报告不声称已完成该最后一步。

08:57:55旧回执的“该条小时投递已处理”已纠正：本轮08:56:59早于08:57:45投递，且原消息仍在官方queue/list。同文提示不能证明具体消息ID被消费。保留原回执及unattended_hourly_processing_reconciliation_20261106.json。

证据：runs/unattended_queue_dispatch_20261106/deployment.json、test_guard/；runs/autonomous_development_20261105/hourly.jsonl、development_status.json。历史桌面缺席执行证据继续保留，未测试物理游戏本断电或轻薄本重启。

## 实际跨轮验收结果

09:09:15–09:09:46暂停所有桌面代理通信；新轮01a1ae67-36dd-75b0-a27c-34f3f97015d9于09:09:22启动，09:09:26首个AI工具observe成功，proxy294825=T且工具来自原生后台13827→1。原生队列自行接续，未发生显式queue/start；观察器未捕获短暂idle，无dispatch_observation文件，不能补造该文件或称卸载线程恢复已实测。09:09:46finally自动恢复，ExecStopPost也正常，两服务成功退出，代理Ssl。

这补齐了真正跨轮、无桌面通信的新AI工具执行；未测物理断电/重启。证据boundary_probe/execution_receipt.json及observation.json。后续继续开发，不重复同验收。
