# task2 - 作业二 - 知乎爬虫

用于爬取知乎 [TYPE - MOON（型月）](https://www.zhihu.com/topic/19840977/hot)话题的部分问题及其答案。  

**【2025.11.01笔记】**  

+ 连续奋战了 7 个小时终于写完了，然而关于 Selenium 可以说我基本没搞懂，基本上只用了`drives.get()`，其他的全靠`drives.page_source`打包给 BeautifulSoup (感觉 Selenium 的查找很诡异，但实际上查找的话用哪个应该都没差？)  

+ 等待策略什么的是真的没搞懂了，遇到报错就直接`while True`加`try-expect`，之后在慢慢学吧。（7 个小时真的燃尽了）