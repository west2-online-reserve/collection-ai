# 李宏毅生成式AI导论 (2025)

## 1.生成式AI基本原理

### token
语言模型理解文字的基本单位是token，token是模型输入输出的基本单位

### 模型
模型可以接受tokens作为输入，然后输出下一个token的概率分布

### 基本原理

将用户输入转为tokens后输入模型，得到下一个token的概率分布，根据概率随机选择概率前top_k的token，并且拼接到用户输入tokens后，然后继续输入模型，直到满足条件停止

```mermaid
flowchart TD
    A[用户输入] --> B[文本转为Token序列]
    B --> C[输入语言模型]
    C --> D[模型输出下一个Token的<br>概率分布]
    D --> E{采样策略}
    E -- Top-K/P采样等 --> F[根据策略选择下一个Token]
    F --> G{新Token是<br>停止符吗?}
    G -- 否 --> H[将新Token拼接到<br>现有序列末尾]
    H -- 循环生成 --> C
    G -- 是 --> I[将最终Token序列<br>转换回文本]
    I --> J[输出最终结果]
```
