# Get Code Skill

名称: `get_code`

目标:
- 根据 `appCode` 获取对应项目的最新代码地址，并拉取/更新最新代码。
- 统一把项目代码存放到 `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo`。
- 任何“读取本地代码 / 查询代码位置 / 提取代码证据”的动作，都必须先完成同轮仓库刷新校验。

可调用方法（code_tool.py）:
1. `clone_repo(git_url: str, repo_root: str | Path = "/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo") -> dict[str, Any]`
2. `pull_repo(git_url: str, repo_root: str | Path = "/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo") -> dict[str, Any]`

方法使用说明:

### 方法1: clone_repo
- 作用: 根据 `git_url` 首次克隆项目到 `src/code_repo/<repo_name>`。
- 参数:
  - `git_url`(必填): 项目仓库地址。
  - `repo_root`(可选): 代码根目录，默认 `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo`。
- 返回:
  - `ok`(bool): 是否成功。
  - `action`(str): 固定为 `clone`。
  - `status`(str): `cloned | already_exists | failed`。
  - `git_url`(str): 输入地址。
  - `target_dir`(str): 本地目标目录。
  - `message`/`stdout`/`stderr`/`return_code`: 执行详情。

### 方法2: pull_repo
- 作用: 根据 `git_url` 在 `src/code_repo/<repo_name>` 更新代码（`git pull --ff-only`）。
- 参数:
  - `git_url`(必填): 项目仓库地址。
  - `repo_root`(可选): 代码根目录，默认 `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo`。
- 返回:
  - `ok`(bool): 是否成功。
  - `action`(str): 固定为 `pull`。
  - `status`(str): `updated | failed`。
  - `git_url`(str): 输入地址。
  - `target_dir`(str): 本地目标目录。
  - `message`/`stdout`/`stderr`/`return_code`: 执行详情。

输入格式:
- 支持单个或多个 `appCode`。
- 建议结构:
```json
{
  "targets": [
    {
      "app_code": "f_tts_trade_order"
    }
  ]
}
```

appCode -> Git 地址映射（示例）:
- `f_tts_trade_order` -> `http://gitlab.corp.qunar.com/flightdev-tts/tts_trade_order.git`
- `f_tts_trade_core` -> `http://gitlab.corp.qunar.com/flightdev-tts/tts-trade-core`

地址解析规则:
1. 优先根据 `app_code` 在映射表/配置中心查询该应用的最新 Git 地址。
2. 查询失败时可使用调用方显式传入的 `git_url` 兜底。
3. 仍无法确定地址时返回失败，并在 `message` 中说明“未找到 appCode 对应代码地址”。

执行规则:
1. 遍历 `targets`。
2. 根据 `app_code` 获取最新 `git_url`。
3. 目标目录固定为:
   - `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo/<app_code>`
4. 如果目标目录已存在并且包含 `.git`，必须执行代码更新:
   - `git -C <dir> pull --ff-only`
5. 如果目标目录不存在，执行首次拉取:
   - `git clone <git_url> <dir>`
6. 失败时返回错误信息，不中断其他仓库处理。

强制校验规则:
1. 只要本轮任务要读取本地代码，必须先执行仓库刷新步骤，禁止直接搜索/读取本地文件。
2. 仓库刷新步骤定义如下:
   - 本地仓库已存在: 必须先执行 `pull_repo` 或 `pull_repo_local`
   - 本地仓库不存在: 必须先执行 `clone_repo`
3. 只有在刷新步骤成功后，才允许继续做文件搜索、代码片段提取、类/方法/行号定位。
4. 如果刷新失败:
   - 本轮代码分析必须立即失败
   - 不得输出“已读取本地代码”“已定位代码位置”这类成功结论
   - 可以保留日志结论，但代码分析部分必须明确写成失败原因
5. `knowledge_lookup` 只能作为代码分析的补充证据，不能替代本该执行的 `pull/clone + 本地代码查询`。

执行顺序模板:
1. 根据 `app_code` 解析 `git_url`
2. 刷新仓库:
   - 已存在则 `pull`
   - 不存在则 `clone`
3. 校验刷新结果为成功
4. 读取本地代码并提取关键证据
5. 输出“代码分析”结果

Planner 可用工具名约定:
- `code_clone`: 使用 `clone_repo`，参数示例 `{"git_url": "<git_url>"}`。
- `code_pull`: 使用 `pull_repo`，参数示例 `{"git_url": "<git_url>"}`。

标准命令:
```bash
mkdir -p /Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo
# 已存在工程时更新最新代码
git -C /Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo/<app_code> pull --ff-only

# 不存在工程时首次克隆
git clone <git_url> /Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo/<app_code>
```

输出约定:
- 返回每个仓库的执行结果:
```json
{
  "results": [
    {
      "app_code": "f_tts_trade_order",
      "git_url": "http://gitlab.corp.qunar.com/flightdev-tts/tts_trade_order.git",
      "target_dir": "/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/code_repo/f_tts_trade_order",
      "status": "cloned | updated | failed",
      "message": "success | error detail"
    }
  ]
}
```
