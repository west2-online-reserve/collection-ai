# 疯狂的一天：我的第一次 GitHub 实验

## 1. 一条 AI 学习之路的开始

今天决定开始学习 AI，感觉这条路会很难，但还是想尝试一下。

> 我觉得学习最难的事就是坚持学习

### 1.1 计算机需要自学

下面是今天看的一段 BFS 代码

```c++
void BFS(int start)
{
    queue<int> q;

    q.push(start);
    visited[start] = true;

    while(!q.empty())
    {
        int u = q.front();
        q.pop();

        cout << u << " ";

        for(int v : graph[u])
        {
            if(!visited[v])
            {
                visited[v] = true;
                q.push(v);
            }
        }
    }
}
```

### 1.2 利用 AI 学习

 现在的 AI 真是太强大了，很多事情都可以辅助完成，也可以帮助我们学习和解决问题。

## 2. 第一次接触 GitHub

下午，我开始学习 Git 和 GitHub。

Git 可以帮助我们记录代码的修改过程，而 GitHub 则可以让我们把代码保存到云端，并与其他人进行协作。

我第一次看到 GitHub 仓库的时候，感觉它有点复杂，有很多我还不懂但是很有用的功能。

### 2.1 我的 Git 学习记录

| 操作    | Git 命令       | 作用            |
| ----- | ------------ | ------------- |
| 初始化仓库 | `git init`   | 创建 Git 仓库     |
| 查看状态  | `git status` | 查看文件修改情况      |
| 添加文件  | `git add`    | 将文件加入暂存区      |
| 提交修改  | `git commit` | 保存一次版本        |
| 上传代码  | `git push`   | 将代码上传到 GitHub |

### 2.2 第一次提交

第一次成功执行 `git push` 的时候，我发现原来把本地代码上传到 GitHub 并没有那么神秘，但是不太熟练。

以后如果我写了一个新的项目，就可以使用 Git 记录代码的变化，再把项目上传到自己的 GitHub 仓库。

---

## 3. 我的第一个 Markdown 文件

为了完成这次作业，我创建了 `crazy-day.md`。

Markdown 最让我感兴趣的是，它不需要复杂的排版软件，只需要使用一些简单的符号，就可以完成标题、列表、代码、链接等内容。

### 3.1 Markdown 学习笔记

常见的 Markdown 语法包括：

* `#`：一级标题
* `##`：二级标题
* `###`：三级标题
* `**文字**`：粗体
* `` `代码` ``：行内代码
* `[文字](链接)`：插入链接
* `![图片描述](图片地址)`：插入图片

如果想进一步了解 Markdown，可以访问 [Markdown Guide](https://www.markdownguide.org/)。

---

## 4. deepseek 的照片

下面放一张 deepseek 的图片

![DeepSeek 图片](https://raw.githubusercontent.com/Neko3000/deepseek-whalechan/refs/heads/main/assets/readme/en/whalechan-hero-banner.webp)

---
