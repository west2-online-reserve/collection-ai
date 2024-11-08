## Transformer
### 语言模型背后的函数——类神经网络
    eg:Transformer(ChatGpt中的T)
### **演变史**
   - N-gram
   - Feed-forward Network
   - Recurrent Neural Network(RNN)
   - Transformer
### **Transformer概述**
#### 得到几率分布的步骤
   - Tokenization（将文字转为Token）
     1.  根据Token List来Tokenization
     2.  通过Byte Pair Encoding（BPE）来找
   
   - Input Layer（理解Token）
     1. Embedding
        1. 将token变为向量（Vector）
     2. Token Embedding是训练时由训练资料找到的
     3. Positional Embedding（位置的资讯加入Embedding）
   - Transformer Block
     1. Attention模组（考虑Token间的关系，理解上下文）
        1. 计算相关性Attention Weight
   

     2. Feed Forward（整合，思考）
   - Output Layer（将Transformer Block的输出转为几率的分布,通常一个Transformer Block就视为一个Layer）
#### 语言模型运作机制
     文字接龙，故只用管和左边的Attention
#### 为什么处理超长文本会是挑战
     计算attention次数和文本长度的平方成正比，故如何加速计算attention是一门学问


## 大型语言模型在想什么RLHF
### 人工智能是黑盒子
   - 不开源or开源程度低（参数，训练过程，训练资料）
   - 不是Interpretable(不是思维透明的)          
      eg:transformer
   - 但并不代表不可解释（可解释性方法）
      1. 我们可以找到影响输出的关键输入(把输入的东西盖住，计算guidian?)
      2. 得到输入输出的关系分析Attention位置
      3. 找出影响输出的关键训练资料，分析embedding中存有什么资讯(probing)
         1. 在BERT中分析，surface--syntactic--semantic过程，但边界模糊
         2. 在训练过程中不断对他做probing
      4. 将embeddinng（高维向量）投影到二维平面上可视化
      5. 语言模型的测谎器（true or false）
      6. 但有更简单方法--直接问eg:信心分数。但提前告诉或暗示语言模型错误的答案，容易误导语言模型，它也可能瞎掰
总结：有两大类解释方法，一个是直接对神经网络进行分析（通常需要一定程度的Transparency）一个是直接问语言模型


## 评估大型语言模型的能力
### 能力方面
   - 将生成的答案与benchmark对比，有选择题形式的benchmark叫MMLU,但语言模型的答案是几率分布，就有不同的评价标准，同时，选项摆放的位置也会影响。
   - 没有单一标准答案的问题类型比如翻译（BLEU），摘要（RUNGE）
   - 人来评比，Chatbot Arena--语言模型的竞技场
   - 用语言模型来评估-MT-bench
   - 大量的各式各样的任务(big bench)...
     1. 用文字冒险游戏，评判语言模型会不会违反道德
     2. 沙莉与小安测验，检测机器有没有心智理论，但举一反三，换个名字与场景，就可能突然没有心智理论，因为语言模型可能之前爬过网上的题目
   -  **注意，要辩证看待语言模型评比结果，换，加一个prompt结果都不一样，eg:cloude，同时也不要尽信benchmark的结果，因为benchmark题目为公开题目，出一些与MMLU相似的考题，提高正确率，或者语言模型早就看过了Benchmark中的资料**
### 其他方面：价格，速度，安全性

## 大型语言模型的各种安全性问题
### 大型语言模型还是会讲错话怎么办
- 有可能产生幻觉，不能把它当作搜索引擎
- 亡羊补牢：事实查核（factcore,facttool），有害词条检测
### 大型语言模型会不会自带偏见
- 检测方法:置换有关词条比如男-女
- 语言模型（红队）负责想一些可能产生偏见的输入，然后reinforcement learning（最大化差距）
- 大型语言模型审查履历可能有偏见，将不同地区的名字的Embedding投影到2维平面上，分布也不一样
- 对职业性别也有刻板印象
- 语言模型也有政治倾向例如比较偏自由主义
- 减轻偏见的方法
  1. 从训练资料介入
  2. 在产生答案的过程中介入（修改模型输出的几率）
  3. 产生答案后，用Post-Processing处理，避免偏见的发生
### 这句话是不是大型语言模型讲的
- 找出人工智能生成的句子和人类生成句子的差别
- 用人工智能来检测：搜集AI生成的句子和人类生成的句子作为训练资料，训练一个分类器AI，但结果不尽人意
- 用AI产生会议审查意见的比例在增加，有趣的是，自然语言处理国际会议，机器学习国际会议首当其冲
- 在语言模型的输出加上水印，简化eg:将所有TOUKEN分成红绿两组，如果今天要产生第奇数个touken,则把绿色touken增加一点概率。如果今天要产生第偶数个touken,则把红组touken增加一点概率。（实际方法更加复杂）值得注意的是，加水印的结果并不会影响产生句子的通顺程度
### 大型语言模型也会被诈骗