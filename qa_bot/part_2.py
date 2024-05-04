from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_transformers import (
    LongContextReorder,
)
from qwen import Qwen
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/project_1/rag.log'),
        logging.StreamHandler()
    ]
)
# 获取对象
logger = logging.getLogger(__name__)

# 配置参数
class RagConfig:
    def __init__(self, query, model_path, vector_model_path, collection_name, db_directory, top_k, rerank_model):
        self.query = query
        self.model_path = model_path
        self.vector_model_path = vector_model_path
        self.collection_name = collection_name
        self.db_directory = db_directory
        self.top_k = top_k
        self.rerank_model = rerank_model

class Rag(object):
    # part_2
    def __init__(self, config):
        self.config = config
        self.llm = Qwen(self.config.model_path)

    def re_query(self):
        """
        传入原始query，返回改写后的
        """
        template = """
                我的问题：
                {query}
                ---
                请根据下列指示改写我的问题：将原始问题转换为更加明确、简洁，并针对性强的形式，以便于在数据库或搜索引擎中进行有效检索。请直接给出改写后的问题。
                """
        prompt = ChatPromptTemplate.from_template(template)

        # 构建 chain
        chain = prompt | self.llm
        res = chain.invoke({"query": self.config.query})

        return res

    def retrieve(self):
        # 检索器实例化
        
        embed_model = HuggingFaceBgeEmbeddings(
            model_name=self.config.vector_model_path,
            model_kwargs= {'device': 'cuda'},
            # 当向量都被规范化（归一化）后，它们的范数都是1。
            # 余弦相似度的计算只需要向量的点积运算，从而减少了计算的复杂度，加快了处理速度。
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="为这个句子生成表示以用于检索相关文章："
            )
        # 从磁盘中加载数据库对象，一定要指明集合名
        vectorstore = Chroma(
                        collection_name=self.config.collection_name,
                        persist_directory=self.config.db_directory, 
                        embedding_function=embed_model
                        )
        # 检查存储成功与否
        # res = vectorstore.get(ids=['1','2'])
        # 直接从数据库创建一个基础检索器
        retriever = vectorstore.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": self.config.top_k,  # 最多十个chunk
                                    # "score_threshold": 1 # 小于等于 
                                    }
                        )
        
        return retriever
        
    def re_write_rank(self, retriever):
        # query改写加召回重排，返回相关文档的字符串
        
        # 改写query
        new_query = self.re_query()
        logger.info(f"改写query：{self.config.query} ———> {new_query}")
        # 余弦相似度召回
        docs = retriever.invoke(new_query)
        # 重排
        tokenizer = AutoTokenizer.from_pretrained(self.config.rerank_model)
        model = AutoModelForSequenceClassification.from_pretrained(self.config.rerank_model)
        model.eval()
        pairs = [[new_query, doc.page_content] for doc in docs]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # 确保scores是一维的
        scores = scores.squeeze()
        # 将docs和scores组合成一个列表
        doc_scores = list(zip(docs, scores.tolist()))
        # 对文档进行排序，基于得分从高到低
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        # 提取排序后的文档对象
        sorted_docs = [doc for doc, score in doc_scores]
        # 长上下文检索器，将最相关的文档放在开始和结尾，中间放无关紧要的
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(sorted_docs)

        return "\n\n".join(doc.page_content for doc in reordered_docs)

    def chat(self):

        retriever = self.retrieve()
        logger.info("加载数据库和检索器成功！")
        context = self.re_write_rank(retriever)
        logger.info(f"改写重排成功！上下文的前50字：{context[:50]}")
        
        PROMPT_TEMPLATE = """参考信息：
            {context}
            ---
            我的问题或指令：
            {question}
            ---
            请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。"""
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        output_parser = StrOutputParser()
        # 构建 chain
        chain = prompt | self.llm | output_parser
        # 生成回复
        res = chain.invoke({"context": context,"question": self.config.query})
        logger.info(f"模型回复：{res}")
        return res


if __name__ == "__main__":


    config = RagConfig(
        query="三年之约是什么意思？",
        model_path="/root/autodl-tmp/Qwen1.5-7B-Chat",
        vector_model_path="/root/project_2/bge-large-zh-v1.5",
        collection_name="db_1",
        db_directory="/root/project_1/chroma_db",
        top_k=10,
        rerank_model="/root/autodl-tmp/bge-reranker-base"
    )

    chat_bot = Rag(config)
    res = chat_bot.chat()
    print(res)


