#coding='utf-8'
#灵海之森
#Python3.6.8
#2022.10.4

import gevent
import requests, time, re  # 发送请求，接收JSON数据，正则解析
from fake_useragent import UserAgent  # 随机请求头
from lxml import etree  # 进行xpath解析
from urllib import parse  # 将中文转换为url编码
import urllib3
import xlwt
import xlrd
urllib3.disable_warnings()


def get_user_info(uid,base_url_1,headers):  # 传入用户id
    #获取用户个人基本信息
    user_data=[]#保存用户信息
    params = {
        'uid': uid,  # 用户id，即uid
    }#参数，用于组配链接
    while True:#防止timeout
        try:
            resp = requests.get(url=base_url_1, params=params, headers=headers,timeout=(30,50),verify=False)
            print(resp)
            break
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue
    data = resp.json()#转换为josn格式
    info=data["data"]["user"]#用户信息
    id=info["id"]#uid
    name=info["screen_name"]#用户名
    verified=info["verified"]#是否认证用户
    if verified == 'TRUE':#只有当是认证用户的时候
        verified_reason=info["verified_reason"]#认证原因/机构
    else:
        verified_reason='未认证'
    location=info["location"]#地理位置
    gender=info["gender"]#性别，f为女，m为男
    followers_count=info["followers_count"]#粉丝数
    statuses_count=info["statuses_count"]#全部微博数
    user_data.append([id,name,verified,verified_reason,location,gender,followers_count,statuses_count])

    return user_data#返回用户基本信息

def get_user_detail_info(uid,base_url_2,headers):  # 传入用户id
    #获取用户个人详细信息
    user_data=[]#保存用户信息
    params = {
        'uid': uid,  # 用户id，即uid
    }#参数，用于组配链接
    while True:#防止timeout
        try:
            resp = requests.get(url=base_url_2, params=params, headers=headers,timeout=(30,50),verify=False)
            #print(resp)
            break
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue
    data = resp.json()#转换为josn格式
    info=data["data"]#用户信息
    birthday=info["birthday"]#birthday
    created_at=info["created_at"]#账号创建时间
    description=info["description"]#简介
    #verified_reason=info["verified_reason"]#认证原因/机构
    try:
        ip_location=info["ip_location"]#ip属地
    except:
        ip_location=info["location"]#使用地点替代ip地址
    
    user_data.append([birthday,created_at,description,ip_location])#

    return user_data#返回用户详细信息


def save_afile(alls,filename):
    """数据保存
        这里是保存单个用户的信息
    """
    f=xlwt.Workbook()
    sheet1=f.add_sheet(u'sheet1',cell_overwrite_ok=True)
    sheet1.write(0,0,'用户id')
    sheet1.write(0,1,'用户名')
    sheet1.write(0,2,'是否认证')
    sheet1.write(0,3,'认证所属机构')
    sheet1.write(0,4,'设定地址')
    sheet1.write(0,5,'性别')
    sheet1.write(0,6,'粉丝数')
    sheet1.write(0,7,'发博数')
    sheet1.write(0,8,'生日')
    sheet1.write(0,9,'账号创建时间')
    sheet1.write(0,10,'简介')
    sheet1.write(0,11,'ip属地')
    i=1
    for all in alls:
        #for data in all:
            for j in range(len(all)):
                sheet1.write(i,j,all[j])
            i=i+1
    f.save(r'用户信息/'+filename+'.xls')

def save_files(alls,filename):
    """数据保存
        以微博正文的角度保存
        多个用户放在一个文件
    """
    f=xlwt.Workbook()
    sheet1=f.add_sheet(u'sheet1',cell_overwrite_ok=True)
    sheet1.write(0,0,'用户id')
    sheet1.write(0,1,'用户名')
    sheet1.write(0,2,'是否认证')
    sheet1.write(0,3,'认证所属机构')
    sheet1.write(0,4,'设定地址')
    sheet1.write(0,5,'性别')
    sheet1.write(0,6,'粉丝数')
    sheet1.write(0,7,'发博数')
    sheet1.write(0,8,'生日')
    sheet1.write(0,9,'账号创建时间')
    sheet1.write(0,10,'简介')
    sheet1.write(0,11,'ip属地')
    i=1
    for all in alls:
        for data in all:
            for j in range(len(data)):
                sheet1.write(i,j,data[j])
            i=i+1
    f.save(r'用户信息/'+filename+'.xls')

def extract(inpath):
    """取出uid数据"""
    data = xlrd.open_workbook(inpath, encoding_override='utf-8')
    table = data.sheets()[0]#选定表
    nrows = table.nrows#获取行号
    ncols = table.ncols#获取列号
    numbers=[]
    for i in range(1, nrows):#第0行为表头
        alldata = table.row_values(i)#循环输出excel表中每一行，即所有数据
        result_1 = alldata[2]#取出表中数据，bid
        numbers.append(str(int(result_1)))
    return numbers

if __name__ == '__main__':

    headers = {
    'authority': 's.weibo.com',
    'method': 'GET',
    'scheme': 'https',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'cache-control': 'no-cache',
    'cookie': '',
    'pragma': 'no-cache',
    'Host':'weibo.com',
    'referer': 'https://weibo.com/',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-site',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': UserAgent().random,
        }#com版本

    
    base_url_1 = "https://weibo.com/ajax/profile/info"#基本信息域名
    base_url_2 = "https://weibo.com/ajax/profile/detail"#详细信息域名
    #uids=['7198559139']#根据uid获取，单个用户
    
    filename=''#微博正文文件，多个用户
    uids = extract(r'微博正文/'+filename+'.xlsx')#多个用户
    print(uids)
    infos=[]
    for uid in uids:
        info=[]
        base_info = get_user_info(uid,base_url_1,headers)
        detail_info = get_user_detail_info(uid,base_url_2,headers)
        for i,j in zip(base_info,detail_info):
            info.append(i+j)#组合成一个列表
        print(info)
        infos.append(info)
        #save_afile(info,uid)#保存单个用户信息
    save_files(infos,filename)#保存多个用户信息
