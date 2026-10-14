# 1.我配置的环境
- 我自己配置的环境是 anaconda 里配置的，不用全局环境，是大一 python 老师带我们配置的，所以暂时还没有遇到什么环境上的问题，遇到的主要是文件循环引用问题，这个是根据范例和问ai解决的,整合成jupyter notebook 就没有这个问题了（但是输出被省略了不少不完整，暂时没解决），只不过要加上from __future__ import annotations来前向字符串引用。
## 2.
- 用的 VScode 右上角的，作为运行当前文件的快捷方式，根据我选定的环境进行程序的运行，这里的pokemon的py文件用play.py运行，换成jupyter notebook一开始我用的是anaconda的环境，然后到网页运行就可以了（就是有点麻烦 要用anaconda prompt打开）,因为在.ipynb文件中，vscode上在输入框中进行输入，不过后面发现是在整个文件底部有输出。