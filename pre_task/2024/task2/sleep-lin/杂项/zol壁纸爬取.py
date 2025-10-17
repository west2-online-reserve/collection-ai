import requests,os,time
from multiprocessing import Pool,cpu_count
from bs4 import BeautifulSoup

class SpiderZol(object):
    """
    抓取手机壁纸 https://sj.zol.com.cn/bizhi/1080x1920/
    """

    def __init__(self, start_page=1, end_page=5):
        # 手机壁纸网址
        self.target = 'http://sj.zol.com.cn/bizhi/1080x1920'
        self.web = 'http://sj.zol.com.cn'
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'}
        self.save_dir = "./爬取的东西/zol_data"  # 保存的目录

        # 列表的地址列表
        self.page_urls = []      #抓取的页面网址
        self.image_urls = []     #抓取的图集网址
        # 需要抓取的列表页起始 及 结束
        self.start_page = start_page
        self.end_page = end_page
        self.mkdir("./爬取的东西")
        self.mkdir(self.save_dir)

    def mkdir(self,save_dir):
        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except Exception as e:
                pass



    def get_page_urls(self):
        """
        某一页的地址格式为
        http://sj.zol.com.cn/bizhi/1080x1920/N.html  N为具体那一页

        start_page: 开始的页数
        end_page: 结束的页数
        """

        self.page_urls = []
        for i in range(self.start_page, self.end_page + 1):
            self.page_urls.append(f"{self.target}/{i}.html")

        print("抓取的列表页如下:")
        print("\n".join(self.page_urls))

    def get_image_urls(self,page_url):
        """
        抓取列表页如
        http://sj.zol.com.cn/bizhi/1080x1920/1.html
        解析出其中所有的图集地址
         <a
            class="pic"
            href="/bizhi/detail_10981_121155.html"
            target="_blank"
            hidefocus="true"
            title = "中国最美公路227国道沿途风光"
        >
        """
        req = requests.get(url=page_url,headers=self.headers)
        html = req.content.decode('GB2312')
        b = BeautifulSoup(html,'lxml')
        a_tags = b.find_all('a',class_='pic')
        for each in a_tags:
            image_name = each.text.strip()
            image_url = self.web+each.get('href','')
            print(f"解析出图集:《{image_name}》",image_url)
            self.image_urls.append({"image_name": image_name, "image_url": image_url})

    #爬取某个图集
    def spider_one_images(self, image_name, image_url, cur_image_no=1):
        """
        image_name: 图集名 可爱吃货油爆叽丁图片壁纸 (9张)
        image_url: 图集地址 http://sj.zol.com.cn/bizhi/detail_10981_121155.html

        如http://sj.zol.com.cn/bizhi/detail_10981_121155.html
        image 标签
        <img
            id="bigImg"
            src="https://sjbz-fd.zol-img.com.cn/t_s320x510c5/g6/M00/0E/07/ChMkKV-NICeIOAUKAEr-VrVkwWAAAD5UQDbYv8ASv5u449.jpg"
            width="320"
            height="510"
            alt=""
        />
        下一页a 标签
         <a
            id="pageNext"
            class ="next"
            href="/bizhi/detail_10981_121153.html" title="点击浏览下一张壁纸，支持'→'翻页"
        >
        """
        # 通过image_name 解析出 可爱吃货油爆叽丁图片壁纸 (9张)  当前图集有多少张图
        max_image_num = int(image_name.split(" ")[-1][1:-2])
        if cur_image_no > max_image_num:
            print("退出已到图集最后一张图")
            return

        req = self.try_request_url(image_url)
        bs = BeautifulSoup(req.text, "lxml")

        print("第{}张图 地址:{}".format(cur_image_no, image_url))
        # 通过 <img id="bigImg"> 来定位解析出图片的真实地址
        for tag in bs.find_all("img", id="bigImg"):
            # 解析出图片地址，并请求下载保存数据
            self.save_image(image_name, tag.get("src"), cur_image_no)

        # 解析下一张图的递归抓取
        for tag in bs.find_all("a", id="pageNext"):
            next_image_url = tag.get("href")
            self.spider_one_images(
                image_name, f"{self.web}{next_image_url}", cur_image_no + 1
            )

    def save_image(self, image_name, image_src, image_no):
        """
        image_name: 图集名 如 可爱吃货油爆叽丁图片壁纸 (9张)
        image_src: 图片下载地址 如 https://b.zol-img.com.cn/sjbizhi/images/11/320x510/160308409840.JPG
        image_no: 当前图片的编号，第几张图
        """
        image_title = image_name.split(" (")[0]  # 去掉` (9张)`作为图集的title
        image_dir = os.path.join(self.save_dir, image_title)
        self.mkdir(image_dir)
        try:
            # 请求图片
            req = self.try_request_url(image_src)
            # 图片最终保存路径
            img_path = os.path.join(image_dir, "{}.jpg".format(image_no))

            # 图片非文本，需要以二进制模式b打开
            with open(img_path, "wb") as f:
                # 使用req.content 获取内容源数据
                f.write(req.content)
                print("{}/{}.jpg 保存成功".format(image_title, image_no))
        except:
            pass
    
    def try_request_url(self, url):
        try_num = 3
        i = 0
        req = None
        while i < try_num:
            req = requests.get(url, headers=self.headers)
            if req.status_code == 200:
                return req
            print(req.status_code, "尝试重新抓取", url)
            print(req.text)
            time.sleep(2)
            i += 1
        return req
    
    def f(self,item):
        print("开始抓取图集:{}".format(item["image_name"]))
        self.spider_one_images(f"{item["image_name"]}",f"{item["image_url"]}",1)

    def go(self):
        start_time = time.time()
        self.get_page_urls()
        for page_url in self.page_urls:
            self.get_image_urls(page_url)
        
        with Pool(cpu_count()) as p:
            p.map(self.f,self.image_urls)
    
        end_time = time.time()
        print(f"共用时{end_time-start_time}秒！")

if __name__ == '__main__':
    spider = SpiderZol(1,1)
    spider.go()