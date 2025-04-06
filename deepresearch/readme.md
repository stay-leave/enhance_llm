# 一个搜索本地知识库（图文论文信息）和网络信息的项目

## 思路
业务需求：舆情领域论文信息的查询，需要出一个调研报告。
过程：查询——查询改写——整体计划撰写——分步执行——每步考虑是否要执行本地搜索，或者网络搜索——完成所有步骤后，决定是否能够满足需求——满足后，根据搜索的信息，撰写报告

## 工程实现
采用提示词方式，调用zhipu-v4-flash
mcp进行工具使用。
工具定义：search_mcp.py 使用装饰器定义了满足mcp格式的工具
```
    [{'type': 'function', 'function': {'name': 'retrieve', 'description': '本地知识库检索', 'parameters': {'properties': {'query': {'title': 'Query', 'type': 'string'}}, 'required': ['query'], 'title': 'retrieveArguments', 'type': 'object'}}}]
```
客户端：client.py 链接工具，定义总体流程。




## 本地知识库
infinity数据库，链接方式如下：
```
    infinity_object = infinity.connect("/root/RAG/database", config_path="/root/RAG/infinity_config/infinity_conf.toml")
    db_object = infinity_object.get_database("paper")
```

## 网络搜索
jina系列：s.jina.ai进行网页搜索，r.jina.ai进行网页内容读取


## 执行结果示例
```
    [04/06/25 17:08:31] INFO     Processing request of type ListToolsRequest                                                              server.py:534

    Connected to server with tools: ['retrieve']

    MCP Client Started!
    Type your queries or 'quit' to exit.

    Query: 超网络的概念
    [04/06/25 17:09:19] INFO     Processing request of type ListToolsRequest                                                              server.py:534
    available_tools:
    [{'type': 'function', 'function': {'name': 'retrieve', 'description': '本地知识库检索', 'parameters': {'properties': {'query': {'title': 'Query', 'type': 'string'}}, 'required': ['query'], 'title': 'retrieveArguments', 'type': 'object'}}}]
```
