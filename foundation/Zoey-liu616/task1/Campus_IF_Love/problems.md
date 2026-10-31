# Campus IF Love 遇到的问题与解决方法

## 1.story_loop() 缩进错误

Problem:存在if else elif多次缩进错误

Solution:重新检查各个判断语句的缩进层级，保证同一组 `if / elif / else` 左侧对齐。

## 2. 输入礼物不方便

Problem：选择送礼物时得输入礼物对应的文字很麻烦而且慢，降低走剧情的兴趣。

Solution:把礼物转为列表，礼物前面加编号，输入时更为便捷。

## 3. 增加了攻略失败机制

Problem: 好感度为负数后仍会进行该角色剧情。

Solution: 增加判断，当好感为负数后可选择该剧情线重新开始或者开启下一剧情线。
