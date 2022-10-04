#coding='utf-8'
#灵海之森
#python3.6.8
#2022.10.4

import requests, os, csv, traceback
from time import sleep
import time
import xlrd
import pandas as pd
import random
from fake_useragent import UserAgent  # 随机请求头
from pandas.core.frame import DataFrame


per_request_sleep_sec = 5
request_timeout = 10

class WeiboCommentSpider(object):
    headers = {
    'authority': 's.weibo.com',
    'method': 'GET',
    'scheme': 'https',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'cache-control': 'no-cache',
    'cookie': '',#不用在这填
    'pragma': 'no-cache',
    'referer': r'https://weibo.com/',
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
    params = {
        'flow':'0',
        'is_reload':'1', 
        'id':'4663893136511324', 
        'is_show_bulletin':'2', 
        'is_mix':'0', 
        'max_id':'0',
        'count':'20', 
        'uid':'6882947163'
     }
    
    comment_folder = '微博评论'#评论文件的文件夹

    def __init__(self, uid=None, id=None, limit=None, cookie=None, child_max_page=5,max_id=0):
        
        self.params['uid'] = uid#转为全局变量
        self.params['max_id'] = max_id#转为全局变量，每一轮初始化为0
        self.headers['referer']='https://weibo.com/'+uid+'/'+id#更新headers

        if id.isdigit()==False:#判断是纯数字的已经失效，直接全部将字母id转为数字id
            #print('检测到不是纯数字')
            id = self.url_to_mid(id)
            self.params['id'] = id
        else:
            #print('检测是纯数字')
            self.params['id'] = id
        if limit:
            self.limit = limit
        self.child_max_page = child_max_page
        if cookie:
                self.headers['cookie'] = cookie
        if not os.path.exists(self.comment_folder):#判断是否存在文件夹，没有就新建
            os.mkdir(self.comment_folder)
        self.result_file = os.path.join(self.comment_folder, str(self.params['id']) + '.csv')#文件保存
        self.got_comments = []#获得的评论
        self.got_comments_num = 0#获得的评论总数
        self.written_comments_num = 0#已写入的评论数

    def parseChild(self, root_comment_id):
        #子评论的爬取
        child_params = {
            'flow':'0',
            'is_reload':'1', 
         'id':'4663893136511324', 
         'is_show_bulletin':'2', 
         'is_mix':'1', 
         'fetch_level':'1',
         'max_id':'0',
         'count':'20', 
         'uid':'6882947163'
         }
        child_params['uid'] = self.params['uid']
        child_params['id'] = root_comment_id
        page = 1
        child_max_page = self.child_max_page
        while True:#开始爬取
            if page > child_max_page:
                print(f"child page up to max_page_limit == {child_max_page}")
                break
            print(f"............ {root_comment_id} child page: {page} .........")
            try:
                response = requests.get('https://weibo.com/ajax/statuses/buildComments', headers=(self.headers), params=child_params,
                  timeout=request_timeout)
                if response.encoding !='utf-8':
                    response.encoding='utf-8'#防止乱码
            except:
                print(traceback.format_exc())
                break
            print(response.encoding)
            res_json = response.json()
            max_id = self.parse(res_json, root_comment_id)#获取翻页指示
            if not max_id:
                break
            child_params['max_id'] = max_id
            sleep(random.randint(5,10))#设置每次爬取的等待时间
            page += 1

    ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #bid（字母数字混合）转换为数字id
    def base62_decode(self,string, alphabet=ALPHABET):
    
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
    def url_to_mid(self,url):
        """
        >>> url_to_mid('z0Ijpwgk7')
        3501703397689247
        """
        url = str(url)[::-1]
        size = len(url) / 4 if len(url) % 4 == 0 else len(url) // 4 + 1
        result = []
        for i in range(size):
            s = url[i * 4: (i + 1) * 4][::-1]
            s = str(self.base62_decode(str(s)))
            s_len = len(s)
            if i < size - 1 and s_len < 7:
                s = (7 - s_len) * '0' + s
            result.append(s)
        result.reverse()
        return int(''.join(result))

    def parse(self, res_json, root_comment_id=None):
        if res_json['ok'] == 0:#响应失败
            print('system is busy')
            return
        else:
            comments = res_json['data']
            #print(comments)
            if len(comments) == 0:#评论数量判断
                print('data crawl finish!')
                return
            for comment in comments:
                comment_id = comment['id']
                comment_time = comment['created_at']
                comment_user_name = comment['user']['screen_name']
                comment_user_link = 'https://weibo.com/' + comment['user']['profile_url']
                comment_user_id=comment['user']['id']#用户id
                comment_content = comment['text']
                try:
                    comment_location=comment['source']#地理位置
                except:
                    print(str(comment_id)+'评论的ip地址获取失败')
                    comment_location='获取失败'
                try:
                    comment_like_num = comment['like_counts']
                except:
                    try:
                        comment_like_num = comment['like_count']
                    except:
                        comment_like_num = 0

                child_comment_num = comment['total_number'] if root_comment_id is None else 0
                one_comment = {
                 '父节点评论id':'' if root_comment_id is None else root_comment_id, #父评论id
                 '当前评论id':comment_id, #评论id
                 '评论时间':comment_time, #时间
                 '评论用户名':comment_user_name, #评论用户名
                 '评论用户id':comment_user_id,
                 '评论用户主页link':comment_user_link, #评论用户链接
                 '评论内容':comment_content, #内容
                 '评论点赞数':comment_like_num, #点赞数
                 '评论ip归属地':comment_location,
                 '子评论个数':child_comment_num#子评论个数
                 }
                one_comment_list = [
                 '' if root_comment_id is None else root_comment_id, 
                 comment_id, 
                 comment_time, 
                 comment_user_name, 
                 comment_user_id,
                 comment_user_link, 
                 comment_content,
                 comment_like_num, 
                 comment_location,
                 child_comment_num
                 ]
                self.got_comments.append(one_comment)
                self.got_comments_num += 1
                #print(self.got_comments_num)
                if not root_comment_id:
                    self.parseChild(comment_id)

            if not res_json['max_id']:
                print('max_id is null')
            return res_json['max_id']

    def save_excel(self,a_list,filename):
        #将所有数据写入excel
        data=DataFrame(a_list)#这时候是以行为标准写入的
        #data=data.T#转置之后得到想要的结果
        data.rename(columns={0:'父节点评论id',1:'当前评论id',2:'评论时间',3:'评论用户名',4:'评论用户id',5:'评论用户主页link',6:'评论内容',7:'评论点赞数',8:'评论ip归属地',9:'子评论个数'},inplace=True)#注意这里0和1都不是字符串
        DataFrame(data).to_excel(r'微博评论/'+str(filename)+'.xlsx',sheet_name='评论',index = False)

    def write_csv(self):
        """将爬取的信息写入csv文件"""
        try:
            result_headers = [
             '父节点评论id',
                 '当前评论id',
                 '评论时间',
                 '评论用户名',
                 '评论用户id',
                 '评论用户主页link',
                 '评论内容',
                 '评论点赞数',
                 '评论ip归属地',
                 '子评论个数'
             ]
            result_data = [w.values() for w in self.got_comments][self.written_comments_num:]
            #print(self.got_comments)
            #print(self.got_comments_num)
            with open((self.result_file), 'a', encoding='utf-8-sig', newline='') as (f):
                #writer = csv.DictWriter(f)
                writer = csv.writer(f)
                if self.written_comments_num == 0:
                    writer.writerows([result_headers])
                writer.writerows(result_data)
            print('%d 条评论写入csv文件完毕:' % self.got_comments_num)
            self.written_comments_num = self.got_comments_num
        except Exception as e:
            print('Error: ', e)
            traceback.print_exc()

    def drop_duplicate(self, path, col_index=0):
        df = pd.read_csv(path,engine='python',encoding='utf-8-sig',error_bad_lines=False)
        first_column = df.columns.tolist()[col_index]
        df.drop_duplicates(keep='first', inplace=True, subset=[first_column])
        df = df[(-df[first_column].isin([first_column]))]
        df.to_csv(path, encoding='utf-8-sig', index=False)

    def crawl(self):
        #爬取程序
        page = 1
        while True:
            print(f"............page: {page} .........")
            try:
                response = requests.get('https://weibo.com/ajax/statuses/buildComments', headers=(self.headers), params=(self.params), 
                  timeout=request_timeout)
                if response.encoding !='utf-8':
                    response.encoding='utf-8'#防止乱码
                print(response.status_code)
                print(response.url)
                print(response.encoding)
                res_json = response.json()
            except:
                #print(traceback.format_exc())
                #break
                print('出错了，暂停一下，稍后重试！')
                sleep(60)
                continue
            
            try:
                self.params['max_id'] = self.parse(res_json)
                max_id = self.params['max_id'] 
            except:
                print(res_json['data'])
                print(traceback.format_exc())
                break

            if not max_id:
                print('未发现翻页指示，共计'+str(self.got_comments_num)+'条评论，退出！')
                break
            if max_id == 0:
                print('爬取完毕，共计'+str(self.got_comments_num)+'条评论，退出！')
                break
            #self.params['max_id'] = max_id
            if page % 3 == 0:#取余数，每3页保存一次
                if self.got_comments_num > self.written_comments_num:
                    self.write_csv()
            if self.written_comments_num >= self.limit:
                break
            sleep(random.randint(8,14))
            page += 1
        print(self.got_comments)
        if self.got_comments_num > self.written_comments_num:
            self.write_csv()
        #self.save_excel(self.got_comments,self.params['id'])
        if os.path.exists(self.result_file):
            self.drop_duplicate((self.result_file), col_index=1)

def extract(inpath):
    """取出bid,uid数据"""
    data = xlrd.open_workbook(inpath, encoding_override='utf-8')
    table = data.sheets()[0]#选定表
    nrows = table.nrows#获取行号
    ncols = table.ncols#获取列号
    numbers=[]
    for i in range(1, nrows):#第0行为表头
        alldata = table.row_values(i)#循环输出excel表中每一行，即所有数据
        result_1 = alldata[1]#取出表中数据，bid
        result_2 = alldata[2]#取出表中数据，uid
        numbers.append([result_1,str(int(result_2))])
    return numbers

def main():
    
    print('微博评论爬取程序开始运行！')
    
    #自己的cookie
    cookie = ''
    child_max_page = 40#子评论最大爬取页数，自己定义，越大数量越多速度越慢
    filename = ''#正文文件名
    config = extract(r'微博正文/'+filename+'.xlsx')#正文文件路径
    for i in config:
        uid = i[1]
        mid = i[0]
        limit = 10000
        print(f"-------- 开始爬取 mid = {mid} --------")
        spider = WeiboCommentSpider(cookie=cookie, uid=uid, id=mid, limit=limit, child_max_page=child_max_page)
        spider.crawl()#开始爬取
        sleep(random.randint(15,25))#随机等待时间，防止400错误




if __name__ == '__main__':
    main()


