# crazy-day
> 疯狂的一天
> 练习 Markdown

## 上午
7 点起床，突发奇想，决定出门

### 早餐
食物如下：

| 名称 | 数量 | 备注 |
| ---- | ---- | ---- |
| 牛奶 | 1盒 | 常温 |
| 面包 | 2片 | 全麦面包 |
| 鸡蛋 | 1个 | 茶叶蛋 |

吃完早餐，打开高德地图，寻找最近的网吧···
网吧图片如下：
![网吧周边](https://mmbiz.qpic.cn/mmbiz_jpg/FrlMmzeTicAAzpTeK1vpQlF69gNNF1M3tqlkFrbichScgiamQfrmsfdeUogMbpBL9B4mNfTPjoS2gH67ibiaQKfBYwg/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

### 网吧
准备打MC，[恋还是练](https://mp.weixin.qq.com/s?__biz=MzA5OTA2ODYzMg==&mid=2672809580&idx=1&sn=6c417c710d331c33dfb39afc7371b2e4&chksm=8bb1a12bf8a8a178c976373c91e8bb6f0fae65169ce2d03190442386bd2f7f7b0a7b426356c4&scene=27)  
然后呢？  
当然是来网吧写个酣畅淋漓的 python 小脚本

```python
def calTime():
    # 计算出门游玩时长
    start_hour = 7
    end_hour = 20
    total_hour = end_hour - start_hour
    print(f"本次出门一共 {total_hour} 小时")
    
if __name__ == "__main__":
    calTime()
```
