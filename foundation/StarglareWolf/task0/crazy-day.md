# 如何将项目git到仓库
## 第一步，打开项目
![右键选择项目-复制路径(或者在文件夹里输入)](step1.png)
## 第二步，进入相应文件夹
![右键选择Git Bash(如果你有下载Git的话)](step2.png)
## 第三步，进入小黑屏后
```bash
git config --global user.email 123@qq.com      #你的邮箱
git config --global user.name zhangsan         #你的用户名
git init                                       #建立一个虚拟仓库
git remote add origin https://what_you_create  #你想放置的仓库地址
git branch -M main
echo "# Hello World" > README.md               #将文本内容覆盖到 readme.md
git add .                                      #添加全部的发生更改的文件
git commit -m "first commit"                   #设置提交信息
git push -u origin main                        #将你的代码 push 到远程仓库(只有执行完这步你才能在仓库中看到更改)
```
这样你就可以将代码上传了
## 我遇到的问题
git remote add origin  时，同一文件夹下仓库选错了，所以我又 remote 了一遍，显示 remote origin already exists.
如果需要更改，就不能再进行 remote 了，需要填写以下代码
```bash
git remote set-url origin https://github.com/用户名/新仓库名.git
#然后检验是否更改完毕
git remote -v
```
这样就可以了

**参考文章**
[如何git](https://west2-online.feishu.cn/wiki/SKQ4wTtZji7Hv4kPquRcJOqEnKg?from=from_copylink)
[pull request](https://west2-online.feishu.cn/wiki/L1ItwNiQEi5ZIik8mIqc7GDQniV?from=from_copylink)