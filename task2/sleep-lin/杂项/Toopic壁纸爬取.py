import requests,os,time
from multiprocessing import Pool,cpu_count
from bs4 import BeautifulSoup

class SpiderToopic(object):
    """
    抓取电脑壁纸 https://www.toopic.cn/
    """
    def __init__(self) -> None:
        self.web = "https://www.toopic.cn"
        self.target = 'https://www.toopic.cn/dnbz/?q=-------.html&px=hot'
        self.image_urls = []
        self.save_dir = './爬取的东西/toopic壁纸'

    def get_image_urls(self,page_url):
        req = self.try_request_url(page_url)
        b = BeautifulSoup(req.content,'lxml')
        a_tags = b.find_all('a',class_='pic')
        for each in a_tags:
            self.image_urls.append(self.web+each.get('href'))

    def spider_images(self):
        with Pool(cpu_count()) as p:
            p.map(self.spider_image,self.image_urls)
    
    def spider_image(self,image_url):
        req = self.try_request_url(image_url)
        b = BeautifulSoup(req.content,'lxml')
        image = b.find('img',alt=True)
        image_name = image.get('alt','')
        image_src = image.get('src','')
        print(f"进程{os.getpid()},处理{image_url}")
        self.mkdir("./爬取的东西")
        self.mkdir(self.save_dir)
        self.save_image(image_name,self.web+image_src)


    def save_image(self,image_name,image_src):
        """
        image_name: 图片名
        image_src: 图片下载地址
        """
        image_title = image_name.strip()
        try:
            req = self.try_request_url(image_src)
            img_path = os.path.join(self.save_dir,f"{image_title}.jpg")

            with open (img_path,'wb') as f:
                f.write(req.content)
                print(f"{image_name}保存成功！")
        except:
            print('保存失败！')

    def mkdir(self,dir):
        if not os.path.exists(dir):
            try: 
                os.mkdir(dir)
            except Exception as e:
                print(e)
        

    def go(self):
        start_time = time.time()
        self.get_image_urls(self.target)
        self.spider_images()
        end_time = time.time()
        print(f'抓取结束,共用时{end_time-start_time}秒！')

    def try_request_url(self,url):
        try_num = 3
        i = 0
        req = None
        while i<try_num:
            req = requests.get(url)
            if req.status_code == 200:
                break
            print(req.status_code,'尝试重新抓取',url)
            print(req.text)
            time.sleep(2)
            i+=1
        return req



if __name__ == '__main__':
    spider = SpiderToopic()
    spider.go()
