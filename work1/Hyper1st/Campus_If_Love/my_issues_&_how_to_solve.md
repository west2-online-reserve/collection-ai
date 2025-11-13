# 遇到的问题 和 解决方法
1. 由于是刚学完面向对象编程 [https://github.com/jackfrued/Python-100-Days/blob/master/Day01-20/19.%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E7%BC%96%E7%A8%8B%E8%BF%9B%E9%98%B6.md]  ，一开始写的时候对各种类的属性和方法及其调用都不是很熟练，很多都是边查笔记边写的.
2. 改写完一段代码（送礼好感度逻辑）后代码报错说我没给affinity参数（之前也没填，好像是我误删了，但没报错我就没管），我就按pycharm的提示去改了，在Game类初始化的地方加上了。后来想不起来为什么当时没报错（我记得当时pycharm也有给我三角形感叹号提示），就去问GPT，结果没找到原因，它倒是叫我去把affinity=0写默认值里。我一看确实更简洁就改了。再后来又去看了看answer，发现原本代码就是这样写的 (+_+)?
3. 拆分代码的时候其实不是很懂怎么分，差点把manage的代码写到game里了。后来还是问GPT叫它帮我梳理了一下就懂了。以下是GPT的原话
    > 📌 拆分步骤（操作清单）  
新建 story.py，把 DIALOGUES / GIFT_EFFECTS / ENDING 移过去。  
在 manage.py 里 from story import DIALOGUES, GIFT_EFFECTS, ENDING。  
保留 Character 和 Game 类在 manage.py。  
新建 game.py，只写启动逻辑。  
运行 python game.py 测试是否正常。  
4. ~~想不起来有啥问题了~~，哦对，md语法也是现学现卖的( ͡• ͜ʖ ͡• ).