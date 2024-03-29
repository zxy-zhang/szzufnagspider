import requests
from bs4 import BeautifulSoup
from os import path
# from wordcloud import WordCloud, ImageColorGenerator
# import jieba.analyse
# import matplotlib.pyplot as plt
# from scipy.misc import imread
import time
from pymongo import MongoClient
import pandas as pd
# from pandas import DataFrame


class HouseSpider:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.zfdb = self.client.zfdb

    session = requests.Session()
    baseUrl = "http://sz.zu.fang.com"

    # 每个区域的url
    urlDir = {
        "不限": "/house/",
        "宝安": "/house-a089/",
        "龙岗": "/house-a090/",
        "南山": "/house-a087/",
        "福田": "/house-a085/",
        "罗湖": "/house-a086/",
        "盐田": "/house-a088/",
        "龙华区": "/house-a013080/",
        "坪山区": "/house-a013081/",
        "光明新区": "/house-a013079/",
        "大鹏新区": "/house-a013082/",
        "惠州": "/house-a013058/",
        "东莞": "/house-a013057/",
        "深圳周边": "/house-a016375/",
    }

    region = "不限"
    page = 100

    # 通过名字获取 url 地址
    def getRegionUrl(self, name="宝安", page=10):
        urlList = []
        for index in range(page):
            if index == 0:
                urlList.append(self.baseUrl + self.urlDir[name])
            else:
                urlList.append(self.baseUrl + self.urlDir[name] + "i3" +
                               str(index + 1) + "/")
        return urlList

    # MongoDB 存储数据结构
    def getRentMsg(self, title, rooms, area, price, address, traffic, region,
                   direction):
        return {
            "title": title,  # 标题
            "rooms": rooms,  # 房间数
            "area": area,  # 平方数
            "price": price,  # 价格
            "address": address,  # 地址
            "traffic": traffic,  # 交通描述
            "region": region,  # 区、（福田区、南山区）
            "direction": direction,  # 房子朝向（朝南、朝南北）
        }

    # 获取数据库 collection
    def getCollection(self, name):
        zfdb = self.zfdb
        if name == "不限":
            return zfdb.rent
        if name == "宝安":
            return zfdb.baoan
        if name == "龙岗":
            return zfdb.longgang
        if name == "南山":
            return zfdb.nanshan
        if name == "福田":
            return zfdb.futian
        if name == "罗湖":
            return zfdb.luohu
        if name == "盐田":
            return zfdb.yantian
        if name == "龙华区":
            return zfdb.longhuaqu
        if name == "坪山区":
            return zfdb.pingshanqu
        if name == "光明新区":
            return zfdb.guangmingxinqu
        if name == "大鹏新区":
            return zfdb.dapengxinqu

    #
    def getAreaList(self):
        return [
            "不限",
            "宝安",
            "龙岗",
            "南山",
            "福田",
            "罗湖",
            "盐田",
            "龙华区",
            "坪山区",
            "光明新区",
            "大鹏新区",
        ]

    def getOnePageData(self, pageUrl, reginon="不限"):
        rent = self.getCollection(self.region)
        self.session.headers.update({
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"
        })
        res = self.session.get(pageUrl)
        
        soup = BeautifulSoup(res.text, "html.parser")
        # 获取需要爬取得 div
        divs = soup.find_all("dd", attrs={"class": "info rel"})
        items = list()
        for div in divs:
            ps = div.find_all("p")
            try:
                # 爬取并存进 MongoDB 数据库
                roomMsg = ps[1].text.split("|")
                # rentMsg 这样处理是因为有些信息未填写完整，导致对象报空
                area = roomMsg[2].strip()[:len(roomMsg[2]) - 2]
                print(area)
                # 标题 房间数 平方数 价格 地址 交通描述 区 房子朝向
                rentMsg = self.getRentMsg(
                    ps[0].text.strip(),
                    roomMsg[1].strip(),
                    int(float(area)),
                    int(ps[len(ps) -
                           1].text.strip()[:len(ps[len(ps) - 1].text.strip()) -
                                           3]),
                    ps[2].text.strip(),
                    ps[3].text.strip(),
                    ps[2].text.strip()[:2],
                    roomMsg[3],
                )
                # 插入到数据库中
                items.append(rentMsg)
            except:
                import traceback;traceback.print_exc()
                continue
        return items

    
    # 设置区域
    def setRegion(self, region):
        self.region = region

    # 设置页数
    def setPage(self, page):
        self.page = page

    def startSpicder(self):
        items = list()
        for url in self.getRegionUrl(self.region, self.page):
            items.extend(self.getOnePageData(url, self.region))
            print("*" * 30 + "one page 分割线" + "*" * 30)
            time.sleep(1)
           # break
        return items        


spider = HouseSpider()
spider.setPage(10)  # 设置爬取页数
items = list()
for i in range(0, 11):
    spider.setRegion(spider.getAreaList()[i])  # 设置爬取区域
    items.extend(spider.startSpicder())  # 开启爬虫
   # break
df = pd.DataFrame(items)
excel_writer = pd.ExcelWriter('rent.xlsx')
df.to_excel(excel_writer)
excel_writer.save()
excel_writer.close()
