from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import chromadb
from langchain_chroma import Chroma
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/project_1/indexing.log'),
        logging.StreamHandler()
    ]
)
# 获取对象
logger = logging.getLogger(__name__)


class Indexing(object):
    # part_1
    def load_pdf(self, path):
        """
        传入pdf路径，返回list，每页为一个元素
        """
        # 是以每页为一个元素的
        loader = PyPDFLoader(path)
        pages = loader.load() # 不管是不是split都是按页
        return pages
    
    def load_txt(self, path):
        """
        传入txt路径，返回list，每页为一个元素
        """
        # 是以每页为一个元素的
        loader = TextLoader(path)
        # split影响返回的元素内容 
        texts = loader.load() # load_and_split()
        return texts

    def split_chunk(self, data, chunk_size, chunk_overlap):
        """
        data: 加载完成的文档
        chunks: 文档的分块
        """
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, # 指定每个文本块的目标大小
                chunk_overlap=chunk_overlap, # 指定文本块之间的重叠字符数
                length_function=len, # 用于测量文本长度的函数，这里使用Python内置的`len`函数。
                is_separator_regex=False, # 指定`separators`中的分隔符是否应被视为正则表达式，这里设置为False，表示分隔符是字面字符。
                separators=["\n\n",  "\n",   " ",    ".",    ",",     "，",  "。", ] # 定义用于分割文本的分隔符列表。
            )
        
        chunks = text_splitter.split_documents(data)

        return chunks
    
    def embedding(self, model_name):
        # 实例化bge向量模型
        embed_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs= {'device': 'cuda'},
            # 当向量都被规范化（归一化）后，它们的范数都是1。
            # 余弦相似度的计算只需要向量的点积运算，从而减少了计算的复杂度，加快了处理速度。
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="为这个句子生成表示以用于检索相关文章："
        )

        return embed_model
    
    def store_vector(self,chunks, embed_model, collection_name, persist_directory):
        # 向量入库
        client_settings = chromadb.config.Settings(
            persist_directory=persist_directory
            )
        client = chromadb.PersistentClient(path=persist_directory) # 数据保存在磁盘
        # 存入数据
        ids = [str(i) for i in range(1, len(chunks) + 1)] # 手动指定id，必须是str
        # 直接从数据生成向量数据库，返回数据库对象
        vectorstore = Chroma.from_documents(
                                            documents=chunks, 
                                            embedding=embed_model, 
                                            ids=ids,
                                            collection_name=collection_name,
                                            collection_metadata={"hnsw:space": "cosine"}, # 默认使用l2距离，支持余弦，内积
                                            persist_directory = persist_directory,
                                            client_settings=client_settings,
                                            client = client
                                            )
        # 不用返回，直接存储到本地数据库了

    
    def main(self, pdf_file, txt_file):
        # 参数
        chunk_size = 300
        chunk_overlap = 50
        model_name = "/root/project_2/bge-large-zh-v1.5" # 向量模型路径
        collection_name = "db_2" # 数据集合名称，非常重要
        persist_directory = "/root/project_1/chroma_db" 


        # 运行
        texts = self.load_txt(txt_file)
        pages = self.load_pdf(pdf_file)
        # 合并list 
        data = texts+pages

        logger.info(f"加载数据")

        chunks = self.split_chunk(data, chunk_size, chunk_overlap)
        logger.info(f"切分数据")
        embed_model = self.embedding(model_name)
        logger.info(f"实例化向量模型")
        store_vector = self.store_vector(chunks, embed_model, collection_name, persist_directory)
        logger.info(f"数据入库")


if __name__ == "__main__":

    pdf_file = "/root/project_1/data/不平衡数据集上在线评论有用性识别研究_刘嘉宇.pdf"
    txt_file = "/root/project_1/data/日志.txt"

    indexing = Indexing()
    indexing.main(pdf_file, txt_file)



# 读取数据

def load_pdf(path):
    """
    传入pdf路径，返回list，每页为一个元素
    """
    # 是以每页为一个元素的
    loader = PyPDFLoader(path)
    pages = loader.load() # 不管是不是split都是按页
    return pages

def load_txt(path):
    """
    传入txt路径，返回list，每页为一个元素
    """
    # 是以每页为一个元素的
    loader = TextLoader(path)
    # split影响返回的元素内容 
    texts = loader.load() # load_and_split()
    return texts
    
texts = load_txt('project_1/data/斗破苍穹.txt')
pages = load_pdf('project_1/data/1.pdf')
# 合并list 
data = texts+pages

# 文本分块
# 实例化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # 指定每个文本块的目标大小
    chunk_overlap=50, # 指定文本块之间的重叠字符数
    length_function=len, # 用于测量文本长度的函数，这里使用Python内置的`len`函数。
    is_separator_regex=False, # 指定`separators`中的分隔符是否应被视为正则表达式，这里设置为False，表示分隔符是字面字符。
    separators=["\n\n",  "\n",   " ",    ".",    ",",     "，",  "。", ] # 定义用于分割文本的分隔符列表。
)
# 分块
chunks = text_splitter.split_documents(data)


# 向量化
# 实例化bge调用
embed_model = HuggingFaceBgeEmbeddings(
    model_name="project_2/bge-large-zh-v1.5",
    model_kwargs= {'device': 'cuda'},
    # 当向量都被规范化（归一化）后，它们的范数都是1。
    # 余弦相似度的计算只需要向量的点积运算，从而减少了计算的复杂度，加快了处理速度。
    encode_kwargs={'normalize_embeddings': True},
    query_instruction="为这个句子生成表示以用于检索相关文章："
)


# 向量数据库
# 客户端,使用该配置
collection_name = 'db_1'
client_settings = chromadb.config.Settings(
        persist_directory='/root/project_1/chroma_db'
)
client = chromadb.PersistentClient(path="/root/project_1/chroma_db") # 数据保存在磁盘
# 引用指定集合，如果集合不存在则自动创建一个新的

# 发现报错InvalidDimensionException: Dimensionality of (1024) does not match index dimensionality (384)
# 把之前的删掉
# client.delete_collection(collection_name)
#collection = client.get_or_create_collection(name=collection_name,metadata={"hnsw:space": "l2"})


# 存入数据
ids = [str(i) for i in range(1, len(chunks) + 1)] # 手动指定id，必须是str
# 直接从数据生成向量数据库，返回数据库对象
vectorstore = Chroma.from_documents(
                                    documents=chunks, 
                                    embedding=embed_model, 
                                    ids=ids,
                                    collection_name=collection_name,
                                    collection_metadata={"hnsw:space": "cosine"}, # 默认使用l2距离，支持余弦，内积
                                    persist_directory = "/root/project_1/chroma_db",
                                    client_settings=client_settings,
                                    client = client
                                    )


# res = vectorstore.get(ids=['1','2'])


# 删掉集合
# Chroma().delete_collection()

'''# 更新数据（示例
chunks[0].metadata = {
    "source": "../../modules/state_of_the_union.txt",
    "new_value": "hello world",
} # page_content是文本内容
vectorstore.update_document(ids[0], texts[0])
# 删除数据
vectorstore.delete(ids=[ids[-1]])'''






