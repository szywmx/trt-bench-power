## trt-bench-power Git 使用说明

本文档说明本项目如何使用 Git 进行版本管理和远程备份（GitHub），以及如何用备份恢复代码。

---

### 1. 本地仓库初始化（已完成一次）

在项目根目录初始化 Git 仓库：

```bash
cd /mnt/sd0/trt-bench-power
git init
```

全局配置提交用户信息（只需执行一次，所有仓库通用）：

```bash
git config --global user.name "你的名字或昵称"
git config --global user.email "你的邮箱@example.com"
```

忽略结果数据目录等非代码文件，在项目根目录有 `.gitignore`（已创建）：

```gitignore
__pycache__/
*.py[cod]
*.log
*.tmp
*.swp
/trt_bench_results/
```

---

### 2. 首次提交与后续提交

**首次提交（已执行过可以忽略）：**

```bash
cd /mnt/sd0/trt-bench-power

git add .                       # 把代码加入暂存区（忽略 .gitignore 中的内容）
git commit -m "initial commit"  # 生成第一个版本
```

**日常开发提交流程：**

1. 修改 / 添加代码。
2. 查看变更：

   ```bash
   git status
   ```

3. 选择要提交的文件：

   ```bash
   git add 路径/文件名      # 精确选择
   # 或
   git add .               # 当前目录所有变更
   ```

4. 提交并写清楚说明：

   ```bash
   git commit -m "本次修改的简要说明"
   ```

---

### 3. 关联 GitHub 远程仓库并推送备份

在 GitHub 上新建一个仓库，例如地址为：

```text
git@github.com:你的用户名/trt-bench-power.git
```

在本地项目目录关联远程，并推送：

```bash
cd /mnt/sd0/trt-bench-power

git remote add origin git@github.com:szywmx/trt-bench-power.git
git branch -M main
git push -u origin main          # 首次推送
```

之后每次本地 `commit` 后，只需执行：

```bash
git push
```

即可把最新代码备份到 GitHub。

---

### 4. 在其他机器上使用或恢复代码

#### 4.1 在新机器上克隆仓库

在任意目录执行：

```bash
git clone git@github.com:你的用户名/trt-bench-power.git
cd trt-bench-power
```

就得到一份完整的代码副本（不包含被忽略的结果数据目录）。

#### 4.2 从远程备份更新本地代码

当你在同一台或另一台机器上已经有本地仓库时，可以用：

```bash
cd /你的/本地/trt-bench-power
git pull
```

把 GitHub 上最新的提交同步到本地。

---

### 5. 回滚 / 使用历史版本

查看提交历史：

```bash
git log --oneline
```

会看到类似：

```text
abcd1234 initial commit
efgh5678 add new feature
...
```

#### 5.1 临时查看某个历史版本

```bash
git checkout 提交ID
```

查看完后回到最新版本：

```bash
git checkout main
```

#### 5.2 把代码恢复到某个历史版本（慎用）

如果你确认要把当前分支“回退”到某次提交，可以使用硬重置（会修改本地文件，请确保重要未提交修改已经备份）：

```bash
git reset --hard 提交ID
git push -f origin main    # 如果需要让远程也回退，需要强推
```

一般日常开发不建议随便 `reset --hard` 和 `push -f`，除非你非常确定自己在做什么。

---

### 6. 推荐的日常使用习惯

- **小步提交**：每完成一小块清晰的修改就 `git add` + `git commit`。
- **经常备份**：在重要改动后执行 `git push`，确保 GitHub 上有最新版本。
- **不提交数据**：把大文件、测试结果、日志等放进 `.gitignore`，本项目的 `trt_bench_results/` 已经被忽略。
- **遇到问题先看 `git status`**：它能告诉你当前有哪些修改、哪些已暂存、当前在什么分支等。


