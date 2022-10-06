# weibo-crawler

全新微博爬虫项目，博文、评论、用户信息一网打尽！

CSDN，52账号：灵海之森

微信公众号：西书北影。欢迎关注！

https://github.com/stay-leave/weibo-public-opinion-analysis
上一个微博爬虫及主题情感分析的项目，是基于手机端的，欢迎star。

本项目基于网页端，只需填入cookie，能够获取到更丰富的字段数据。cookie获取方法在文末。

包括三个程序，weibo_blog是微博博文搜索，模仿高级搜索，可自定义微博类型、时间段等；

![image](https://user-images.githubusercontent.com/58450966/193724601-98e2b0c6-21e4-4201-944d-a547e426d05c.png)



user_info是微博用户信息爬取，根据博文或评论结果进行爬取，字段属性丰富，包括基本信息和详细信息。可以根据需要爬取单个用户，或者多个。

![image](https://user-images.githubusercontent.com/58450966/193724744-9a78ac95-133b-4f42-9653-c8ff99782965.png)


特别鸣谢：

https://github.com/dataabc/weibo-search

感谢大佬的开源！


cookie获取教程：

1.进入https://weibo.com/

2.F12进入调试页面

3.ctrl+r刷新页面，选择xhr。

4.随便点击一个文件，查看消息头。

5.往下翻到消息头，里面的cookie即是。
