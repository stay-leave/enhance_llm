# weibo-crawler

全新微博爬虫项目，博文、评论、用户信息一网打尽！

微信公众号：西书北影。欢迎关注！

https://github.com/stay-leave/weibo-public-opinion-analysis，上一个微博爬虫及主题情感分析的项目，是基于手机端的，欢迎star。

本项目基于网页端，能够获取到更丰富的字段数据。

包括三个程序，weibo_blog是微博博文搜索，模仿高级搜索，可自定义微博类型、时间段等；

![image](https://user-images.githubusercontent.com/58450966/193724601-98e2b0c6-21e4-4201-944d-a547e426d05c.png)


weibo_comment是微博评论爬取，根据博文结果进行爬取，博文若只有几百条评论基本可以全部获取，但是几千条以上获取率大幅降低；

![image](https://user-images.githubusercontent.com/58450966/193724682-886fb42a-ed2a-40e3-b787-ccfa418b9596.png)


user_info是微博用户信息爬取，根据博文或评论结果进行爬取，字段属性丰富，包括基本信息和详细信息。可以根据需要爬取单个用户，或者多个。

![image](https://user-images.githubusercontent.com/58450966/193724744-9a78ac95-133b-4f42-9653-c8ff99782965.png)


特别鸣谢：https://github.com/dataabc/weibo-search，感谢大佬的开源！
