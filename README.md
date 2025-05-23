微信公众号：西书北影。



## 项目介绍
本项目包含四部分：

1.向量模型垂域微调：基于llama_index和qwen微调BGE向量模型。[http://t.csdnimg.cn/vSmRW](https://blog.csdn.net/qq_43814415/article/details/138358371)

2.大模型垂域微调：基于PEFT微调qwen1.5-7b-chat，做了sft和dpo。[http://t.csdnimg.cn/ndZ47](https://blog.csdn.net/qq_43814415/article/details/138371885)

3.高阶检索增强生成(RAG)系统：基于以上垂域化工作，实现两阶段的RAG系统。增加了query改写、召回重排、检索重排、多轮对话等。[http://t.csdnimg.cn/6nw4D](https://blog.csdn.net/qq_43814415/article/details/138403394)

4.多模态大模型实现：基于qwen2和clip，使用MLP作为连接器，使得语言模型能懂图像。[http://t.csdnimg.cn/9kDTy](https://blog.csdn.net/qq_43814415/article/details/140137599)

5.deepresearch: 基于MCP工具开发的deepresearch的demo。支持同时检索本地知识库和网络信息。

## 硬件配置

显卡：L20(48GB) * 1 
内存：100GB

## 环境配置

1部分的Python环境配置是：
```
cd enhance_llm/ebedding_finetune
pip install -r requirements.txt
```

2.3部分的Python环境配置是：
```
cd enhance_llm
pip install -r requirements.txt
```



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=stay-leave/enhance_llm&type=Date)](https://star-history.com/#stay-leave/enhance_llm&Date)


## LICENSE

enhance_llm is licensed under Apache 2.0 License


