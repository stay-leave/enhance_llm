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
from bs4 import BeautifulSoup#解析网页
import xlwt
import xlrd
from pandas.core.frame import DataFrame
urllib3.disable_warnings()

#微博博文爬虫，使用高级搜索
#数字id转换为bid（字母数字混合）
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
def base62_encode(num, alphabet=ALPHABET):
    """Encode a number in Base X
    `num`: The number to encode
    `alphabet`: The alphabet to use for encoding
    """
    if (num == 0):
        return alphabet[0]
    arr = []
    base = len(alphabet)
    while num:
        rem = num % base
        num = num // base
        arr.append(alphabet[rem])
    arr.reverse()
    return ''.join(arr)
def mid_to_url(midint):
    """
    >>> mid_to_url(3491700092079471)
    'yCtxn8IXR'
    >>> mid_to_url(3486913690606804)
    'yAt1n2xRa'
    """
    midint = str(midint)[::-1]
    size = len(midint) % 7 if len(midint) % 7 == 0 else len(midint) % 7 + 1
    result = []
    for i in range(size):
        s = midint[i * 7: (i + 1) * 7][::-1]
        s = base62_encode(int(s))
        s_len = len(s)
        if i < size - 1 and len(s) < 4:
            s = '0' * (4 - s_len) + s
        result.append(s)
    result.reverse()
    return ''.join(result)
#bid（字母数字混合）转换为数字id
def base62_decode(string, alphabet=ALPHABET):
    """Decode a Base X encoded string into the number
    Arguments:
    - `string`: The encoded string
    - `alphabet`: The alphabet to use for encoding
    """
    base = len(alphabet)
    strlen = len(string)
    num = 0
 
    idx = 0
    for char in string:
        power = (strlen - (idx + 1))
        num += alphabet.index(char) * (base ** power)
        # print('n:', power, num)
        idx += 1
 
    return num
def url_to_mid(url):
    """
    >>> url_to_mid('z0JH2lOMb')
    3501756485200075L
    >>> url_to_mid('z0Ijpwgk7')
    3501703397689247L
    """
    url = str(url)[::-1]
    size = len(url) / 4 if len(url) % 4 == 0 else len(url) // 4 + 1
    result = []
    for i in range(size):
        s = url[i * 4: (i + 1) * 4][::-1]
        s = str(base62_decode(str(s)))
        s_len = len(s)
        if i < size - 1 and s_len < 7:
            s = (7 - s_len) * '0' + s
        result.append(s)
    result.reverse()
    return int(''.join(result))



def get_page(url,headers):  
    #获取搜索出的页数，用于翻页
    
    while True:#防止timeout
        try:
            resp = requests.get(url=url, headers=headers,timeout=(30,50),verify=False)#params=params,
            #resp.encoding=resp.apparent_encoding#使用备用编码
            if resp.encoding !='UTF-8':
                resp.encoding='UTF-8'#防止乱码
            print('当前访问网址：'+str(url))
            print('状态码：'+str(resp.status_code))
            break
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(10)
            print("Was a nice sleep, now let me continue...")
            continue
    html=resp.text#网页
    soup = BeautifulSoup(html,'lxml')#使用bs4解析网页
    #页数列表
    try:
        pages=soup.select('.s-scroll')
        for p in pages:
            page=p#bs4.element.Tag类型
        page=str(page)#str
        page=re.findall('第\d+页',page,re.S)
        page=len(page)#页数
    except:
        page=1#如果只有一页，会获取不到页数
 
    return page

def get_blogs(url,headers):  
    #获取单页的博文信息
    
    while True:#防止timeout
        try:
            resp = requests.get(url=url, headers=headers,timeout=(30,50),verify=False)#params=params,
            #resp.encoding=resp.apparent_encoding#使用备用编码
            if resp.encoding !='UTF-8':
                resp.encoding='UTF-8'#防止乱码
            print('当前访问网址：'+str(url))
            print('状态码：'+str(resp.status_code))
            time.sleep(5)
            break
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue
    #print(resp)
    #print(resp.text)
    #print(resp.headers)
    html=resp.text#网页
    soup = BeautifulSoup(html,'lxml')#使用bs4解析网页
    blogs=soup.select('#pl_feedlist_index > div:nth-child(2)')#用css选择器，选择博文列表
    #print(blogs)#bs4.element.Result类型
    for b in blogs:
        blog=b#bs4.element.Tag类型
    blog=str(blog)#str类型
    blogs_l=re.findall('<!--card-wrap-->(.*?)<!--/card-wrap-->',blog,re.S)#成功获取博文列表的元素，列表形式
    #print(len(blogs_l))#博文有多少，用于判断本页有多少博文
    blogs_number=len(blogs_l)#本页博文数量

    #存储单页的各个属性的数据
    id_data=[]
    bid_data=[]
    uid_data=[]
    name_data=[]
    time_data=[]
    text_data=[]
    repost_data=[]
    comment_data=[]
    like_data=[]

#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div > div.card-act > ul > li:nth-child(4) > a
#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div > div.card-act > ul > li:nth-child(4) > a > em
#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(4) > a:nth-child(1) > button:nth-child(1) > span:nth-child(2)

    #获取单页博文的详细信息
    for i in range(1, blogs_number+1):#如果是第一个，就保留第一个的选择器，如果第2及之后，就换
        if i==1:
            blog_ids=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1)')#数字id
            user_infos=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > a:nth-child(1)')
            #publish_times = soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2) > a:nth-child(1)')
            publish_times=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div > div.card-feed > div.content > p.from > a:nth-child(1)')
            #blog_texts=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(3)')
            blog_texts=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)')
            #reposts=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(1) > a:nth-child(1)')
            reposts=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(2) > a:nth-child(1)')
            #comments=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(2) > a:nth-child(1)')
            comments = soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(3) > a:nth-child(1)')
            #like_counts=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(4) > a:nth-child(1) > button:nth-child(1) > span:nth-child(2)')
            like_counts=soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(4) > a:nth-child(1) > em')
        else: 
            blog_ids=soup.select('div.card-wrap:nth-child('+str(i)+')')
            user_infos=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > a:nth-child(1)')
            #publish_times=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2) > a:nth-child(1)')
            publish_times = soup.select('div.card-wrap:nth-child('+str(i)+') > div > div.card-feed > div.content > p.from > a:nth-child(1)')
            #blog_texts=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(3)')
            blog_texts=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)')
            #reposts=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(1) > a:nth-child(1)')
            reposts=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(2) > a:nth-child(1)')
            #comments=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(2) > a:nth-child(1)')
            comments=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(3) > a:nth-child(1)')
            #like_counts=soup.select('div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(3) > a:nth-child(1) > button:nth-child(1) > span:nth-child(2)')
            like_counts=soup.select('div.card-wrap:nth-child('+str(i)+') > div.card > div.card-act > ul > li:nth-child(4) > a > em')

#pl_feedlist_index > div:nth-child(1) > div:nth-child(4) > div.card > div.card-act > ul > li:nth-child(4) > a > em

            #div.card-wrap:nth-child('+str(i)+') > div:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(1) > a:nth-child(1)
        #博文id、bid提取

        for b in blog_ids:
            blog_id=b#bs4.element.Tag类型
            blog_id=str(blog_id)#str类型
            blog_id=re.findall('<div action-type="feed_list_item" class="card-wrap" mid="(.*?)">',blog_id,re.S)#提取博文id，如['4809724744696284']
            blog_id=[int(id) for id in blog_id]
            blog_bid=[str(mid_to_url(id)) for id in blog_id]
            id_data += blog_id
            bid_data += blog_bid

        #用户信息提取
        for u in user_infos:
            user_info=u#bs4.element.Tag类型
            user_info=str(user_info)#str类型
            uids=re.findall('<a class="name" href="//weibo.com/(.*?)?refer_flag=.*?',user_info)#析取出uid
            uid=[uid.replace('?','') for uid in uids]#如：['1888116862']
            name=re.findall('<a class=.*?nick-name="(.*?)" suda-data=.*?',user_info)#析取出name，如：['投鱼问道']
            uid_data += uid
            name_data += name
            #print(uid)

        #发布时间提取
        # for p in publish_times:
        #     publish_time=p#bs4.element.Tag类型
        # publish_time=str(publish_time)#str 
        # publish_time=re.findall('<a href="//weibo.com.*?>(.*?)</a>',publish_time,re.S)#提取时间，如['09月03日 21:20']
        # publish_time=[publish.strip() for publish in publish_time]#去除前后空白
        # time_data += publish_time
        for p in publish_times:
            publish_time=p#bs4.element.Tag类型
            publish_time=str(publish_time)#str 
            publish_time=re.findall('<a href="//weibo.com.*?>(.*?)</a>',publish_time,re.S)#提取时间，如['09月03日 21:20']
            publish_time=[publish.strip() for publish in publish_time]#去除前后空白
            time_data += publish_time

        #博文文本提取
        # for b in blog_texts:
        #     blog_text=b#bs4.element.Tag类型
        # blog_text=str(blog_text)#str 
        # blog_text=re.findall('<p class="txt".*?>(.*?)</p>',blog_text,re.S)#提取文本
        # blog_text=[text.strip().strip('\u200b').replace(' ','') for text in blog_text]#去空格、非法字符
        # text_data += blog_text
        for b in blog_texts:
            blog_text=b#bs4.element.Tag类型
            blog_text=str(blog_text)#str 
            blog_text=re.findall('<p class="txt".*?>(.*?)</p>',blog_text,re.S)#提取文本
            blog_text=[text.strip().strip('\u200b').replace(' ','') for text in blog_text]#去空格、非法字符
            text_data += blog_text

        
        #转发数提取
        for r in reposts:
            repost=r#bs4.element.Tag类型
            repost=str(repost)#str
            #repost=re.findall('<a.*?</span> (.*?)</a>',repost,re.S)#提取转发数
            repost = re.findall('<a.*?>(.*?)</a>',repost, re.S)  # 提取转发数
            new_repost=[]
            for r in repost:#处理无转发的情况，转发数转为int
                if r.strip() == '转发':
                    r=0
                else:
                    #r=int(r)
                    r = re.findall('[0-9]+', r, re.S)  # 提取评论数
                    #print(r)
                    r=int(r[0])
                new_repost.append(r)
            repost_data += new_repost

        #评论数提取
        #print(str(comments))
        for c in comments:
            comment=c#bs4.element.Tag类型
            comment=str(comment)#str 
            #print("comment:"+comment)
            #comment=re.findall('<a.*?</span> (.*?)</a>',comment,re.S)#提取评论数
            comment=re.findall('<a.*?>(.*?)</a>',comment,re.S)#提取评论数
            #print(comment)
            new_comment=[]
            for c in comment:#处理无评论的情况，评论数转为int
                if c.strip() == '评论':
                    c=0
                else:
                    #int(c)
                    c = re.findall('[0-9]+', c, re.S)  # 提取评论数
                    #print(c)
                    c=int(c[0])
                new_comment.append(c)
            comment_data += new_comment

        #点赞数提取
        # for l in like_counts:
        #     like_count=l#bs4.element.Tag类型
        # like_count=str(like_count)#str 
        # #like_count=re.findall(r'\b\d+\b',like_count,re.S)#匹配纯数字
        # like_count_1=re.findall('<span class="woo-like-count">(.*?)</span>',like_count,re.S)#匹配点赞内容
        # if like_count_1 == []:#处理第一个博文的点赞规则和之后不一样的问题
        #     like_count_1=re.findall('</span> (.*?)</a>',like_count,re.S)
        # new_like_count=[]
        # for l in like_count_1:#处理无点赞的情况，点赞数转为int
        #     if l.strip() == '赞':
        #         l=0
        #     else:
        #         l=int(l)
        #     new_like_count.append(l)
        # like_data += new_like_count
        # time.sleep(1)
        #print(str(like_counts))
        for l in like_counts:
            like_count=l#bs4.element.Tag类型
            like_count=str(like_count)#str 
            # print(str(like_count))
            # print("====")
            #like_count=re.findall(r'\b\d+\b',like_count,re.S)#匹配纯数字
            like_count_1=re.findall('<em>(.*?)</em>',like_count,re.S)#匹配点赞内容
            # print(like_count_1)
            # print("====like_count_1====")
            # if like_count_1 == []:#处理第一个博文的点赞规则和之后不一样的问题
            #     like_count_1=re.findall('</span> (.*?)</a>',like_count,re.S)
            new_like_count=[]
            for l in like_count_1:#处理无点赞的情况，点赞数转为int
                if l.strip() == '':
                    l=0
                else:
                    l = re.findall('[0-9]+', l, re.S)  # 提取评论数
                    # print(l)
                    l=int(l[0])
                new_like_count.append(l)
            like_data += new_like_count
            time.sleep(1)
    
    return id_data,bid_data,uid_data,name_data,time_data,text_data,repost_data,comment_data,like_data#返回单页的数据


def save_excel(a_list,filename):
    #将所有数据写入excel
    data=DataFrame(a_list)#这时候是以行为标准写入的
    data=data.T#转置之后得到想要的结果
    data.rename(columns={0:'博文id',1:'博文bid',2:'用户id',3:'用户名',4:'发博时间',5:'博文文本',6:'转发数',7:'评论数',8:'点赞数'},inplace=True)#注意这里0和1都不是字符串
    DataFrame(data).to_excel(r'微博正文/'+filename+'.xlsx',sheet_name='正文',index = False)

if __name__ == '__main__':

    headers = {
                        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
'Accept-Encoding':'gzip, deflate, br',
'Accept-Language':'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
'Connection':'keep-alive',

'Cookie':'',#填入自己的
'Host':'s.weibo.com',
'Referer':'https://weibo.com/',
'Sec-Fetch-Dest':'document',
'Sec-Fetch-Mode':'navigate',
'Sec-Fetch-Site':'none',
'Sec-Fetch-User':'?1',
'TE':'trailers',
'Upgrade-Insecure-Requests':'1',
'user-agent': UserAgent().random,#随机浏览器标识
                    }#搜索用的headers


    #弄一个列表，满足条件就传入参数及值。前置限制条件
    parameters_one=['&typeall=1','&xsort=hot','&scope=ori','&vip=1','&category=4','&viewpoint=1',]#一级条件全选，热门，原创，认证用户，媒体，观点
    parameters_two=['&suball=1','&haspic=1','&hasvideo=1','&hasmusic=1','&haslink=1',]#二级条件全选，含图片，含视频，含音乐，含短链

    #需要自定义
    one_type=0#微博类型，一级条件筛选。0为全选，1为热门，2为原创，3为认证用户，4为媒体，5为观点
    two_type=0#细分类型，二级条件筛选。0为全选，1为含图片，2为含视频，3为含音乐，4为含短链接
    timescope='2022-12-21-0:2023-01-07-0'#开始时间：结束时间，年-月-日-时，其中时是从0到23
    query='2023新年贺词'#关键词或话题
    #query=parse.quote(query)#将搜索关键词转换编码

    #存储单次搜索的所有数据
    id_datas=[]
    bid_datas=[]
    uid_datas=[]
    name_datas=[]
    time_datas=[]
    text_datas=[]
    repost_datas=[]
    comment_datas=[]
    like_datas=[]

    #最终组配出搜索链接，搜索最多50页，需要找到终止条件
    base_url='https://s.weibo.com/weibo?q='+query+parameters_one[one_type]+parameters_two[two_type]+'&timescope=custom:'+timescope+'&Refer=g'
    page_number=get_page(base_url,headers)#页数
    print('搜索关键词"'+query+'"在'+timescope+'共计'+str(page_number)+'页。')
    time.sleep(3)
    for page in range(1,page_number+1):#开始爬取
        id_data,bid_data,uid_data,name_data,time_data,text_data,repost_data,comment_data,like_data=get_blogs(base_url+'&page='+str(page),headers)#翻页操作
        #将每页的数据都添加进一个列表，元素为最小单位
        id_datas += id_data
        bid_datas += bid_data
        uid_datas += uid_data
        name_datas += name_data
        time_datas += time_data
        text_datas += text_data
        repost_datas += repost_data
        comment_datas += comment_data
        like_datas += like_data
        print('正在爬取第'+str(page)+'页，本页共'+str(len(uid_data))+'条数据。')
        time.sleep(10)
    #将所有结果聚合为一个大列表
    print('所有页的数据都已保存完毕，共计'+str(len(uid_datas))+'条数据。')
    a_list=[id_datas,bid_datas,uid_datas,name_datas,time_datas,text_datas,repost_datas,comment_datas,like_datas]
    save_excel(a_list,query)
    print('文件名为"'+query+'"的文件已保存！')



