import infinity_embedded as infinity
from infinity_embedded.common import ConflictType
from transformers import AutoModel,AutoTokenizer, AutoModelForSequenceClassification
import torch
import base64
from PIL import Image
from io import BytesIO
from harvesttext import HarvestText
import jieba
import pandas as pd
import re
import numpy as np


# 还原多模态chunk内容
def restore_chunk_content(chunk_content, image_res, table_res, formula_res):
    """
    将 chunk_content 中的占位符替换为实际的多媒体内容。

    参数：
    - chunk_content (str): 包含占位符的块内容。
    - image_res (pd.DataFrame): 包含该块所有图片的 DataFrame，必须包含 'images_placeholder' 和 'image_path' 列。
    - table_res (pd.DataFrame): 包含该块所有表格的 DataFrame，必须包含 'chunk_tables_placeholder' 和 'table_content' 列。
    - formula_res (pd.DataFrame): 包含该块所有公式的 DataFrame，必须包含 'chunk_formulas_placeholder' 和 'formula_content' 列。

    返回：
    - str: 替换后的块内容。
    """
    # 创建一个占位符到替换内容的映射字典
    placeholder_mapping = {}
    
    # 处理图片占位符
    for _, row in image_res.iterrows():
        placeholder = row['images_placeholder']
        image_path = row['image_path']
        # 使用 Markdown 格式插入图片
        replacement = f"![Image]({image_path})"
        placeholder_mapping[placeholder] = replacement
    
    # 处理公式占位符
    for _, row in formula_res.iterrows():
        placeholder = row['chunk_formulas_placeholder']
        formula_content = row['formula_content']
        # 使用 LaTeX 语法插入公式
        replacement = f"$$ {formula_content} $$"
        placeholder_mapping[placeholder] = replacement
    
    # 处理表格占位符
    for _, row in table_res.iterrows():
        placeholder = row['chunk_tables_placeholder']
        table_content = row['table_content']
        # 假设表格内容已经是 Markdown 格式，如果不是，可以根据需要调整
        replacement = f"| {table_content} |"
        placeholder_mapping[placeholder] = replacement
    
    # 替换 chunk_content 中的占位符
    for placeholder, replacement in placeholder_mapping.items():
        chunk_content = chunk_content.replace(placeholder, replacement)
    
    # 替换未找到的占位符为默认文本
    chunk_content = re.sub(r'\[(FORMULA|IMAGE|TABLE)_\d+\]', r'[\1_NOT_FOUND]', chunk_content)
    
    return chunk_content

# 还原所有的多模态内容
def restore_all_chunks(chunk_res, image_res, table_res, formula_res):
    """
    对所有块进行还原，替换占位符为实际的多媒体内容。

    参数：
    - chunk_res (pd.DataFrame): 包含所有块的 DataFrame，必须包含 'chunk_uuid' 和 'chunk_content' 列。
    - image_res (pd.DataFrame): 包含所有图片的 DataFrame，必须包含 'chunk_uuid', 'images_placeholder', 'image_path' 列。
    - table_res (pd.DataFrame): 包含所有表格的 DataFrame，必须包含 'chunk_uuid', 'chunk_tables_placeholder', 'table_content' 列。
    - formula_res (pd.DataFrame): 包含所有公式的 DataFrame，必须包含 'chunk_uuid', 'chunk_formulas_placeholder', 'formula_content' 列。

    返回：
    - pd.DataFrame: 包含还原后内容的 DataFrame，新增加 'restored_content' 列。
    """
    # 创建一个副本以避免修改原始数据
    restored_chunk_res = chunk_res.copy()
    restored_chunk_res['restored_content'] = restored_chunk_res['chunk_content']
    
    # 遍历每个块
    for idx, row in restored_chunk_res.iterrows():
        chunk_uuid = row['chunk_uuid']
        chunk_content = row['chunk_content']
        
        # 获取该块的所有图片、表格、公式
        img_res = image_res[image_res['chunk_uuid'] == chunk_uuid]
        tbl_res = table_res[table_res['chunk_uuid'] == chunk_uuid]
        frm_res = formula_res[formula_res['chunk_uuid'] == chunk_uuid]
        
        # 调用替换函数
        restored_content = restore_chunk_content(chunk_content, img_res, tbl_res, frm_res)
        
        # 更新 DataFrame
        restored_chunk_res.at[idx, 'restored_content'] = restored_content

        # 根据 'chapter_uuid' 和 'chunk_order' 进行排序，使同章节的在一起，按顺序
        restored_chunk_res_sorted = restored_chunk_res.sort_values(by=["chapter_uuid", "chunk_order"])
        
        # 重置索引（可选）
        restored_chunk_res_sorted.reset_index(drop=True, inplace=True)
    
    return restored_chunk_res

# query分词
def query_words(query):
    # 加载自定义词表
    jieba.load_userdict("RAG/run/user_dict.txt")
    # 加载停用词表
    def load_stopwords(file_path):
        stopwords = set()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords.add(line.strip())  # 去掉每行的换行符和空格
        return stopwords
    
    stopwords = load_stopwords('RAG/run/stop_words.txt')
    words = jieba.cut(query)
    filtered_words = [word for word in words if word not in stopwords]

    return filtered_words
# chunk级别：根据query进行混合检索
def chunk_hybid_search(db_object, model, query, k=6, threshold=0.0):
    """
    输入：数据库，向量模型，查询，top_K，阈值
    输出：混合检索后的df
    """

    table = db_object.get_table("table_chunk")
    content_embed = model.encode_text(query, task='retrieval.query') # 嵌入
    words = query_words(query) # 分词
    # 混合检索,chunk
    chunk_res = table.output(["chunk_uuid", "paper_uuid", "chapter_uuid", "chunk_order", "chunk_content", "_score"]
                    ).match_text("chunk_content", " ".join(words), k
                    ).match_dense("chunk_embed", content_embed, "float", "cosine", k, {"threshold": str(threshold)}
                    ).fusion("weighted_sum", k, {"weights": "0.8,1.2"}
                    ).to_df()
    
    # 检索多媒体表中，是否有属于该chunk的
    chunk_uuids = chunk_res["chunk_uuid"].tolist()
    # 图片表
    table_image = db_object.get_table("table_image")
    image_res = table_image.output(["image_uuid", "chunk_uuid", "images_placeholder", "image_path"]
                    ).filter(f"chunk_uuid in {tuple(chunk_uuids)}"
                    ).to_df()
    # 表格表
    table_table = db_object.get_table("table_table")
    table_res = table_table.output(["table_uuid", "chunk_uuid", "chunk_tables_placeholder", "table_content"]
                    ).filter(f"chunk_uuid in {tuple(chunk_uuids)}"
                    ).to_df()
    # 公式表
    table_formula = db_object.get_table("table_formula")
    formula_res = table_formula.output(["formula_uuid", "chunk_uuid", "chunk_formulas_placeholder", "formula_content"]
                    ).filter(f"chunk_uuid in {tuple(chunk_uuids)}"
                    ).to_df()
    
    # 还原所有chunk 到字段 restored_content。按章节和内部顺序组织
    restored_chunk_res = restore_all_chunks(chunk_res, image_res, table_res, formula_res)
    # 返回chunk的图片和表格
    
    return restored_chunk_res,image_res,table_res

# chunk级别：根据query进行混合检索，限定在指定paper里
def chunk_hybid_search_with_paper(db_object, model, query, paper_uuids, k=20, threshold=0.0):
    """
    输入：数据库，向量模型，查询，top_K，阈值
    输出：混合检索后的df
    """

    table = db_object.get_table("table_chunk")
    content_embed = model.encode_text(query, task='retrieval.query') # 嵌入
    words = query_words(query) # 分词
    # 混合检索,chunk
    chunk_res = table.output(["chunk_uuid", "paper_uuid", "chapter_uuid", "chunk_order", "chunk_content", "_score"]
                    ).filter(f"paper_uuid in {tuple(paper_uuids)}"   
                    ).match_text("chunk_content", " ".join(words), k
                    ).match_dense("chunk_embed", content_embed, "float", "cosine", k, {"threshold": str(threshold)}
                    ).fusion("weighted_sum", 3, {"weights": "0.8,1.2"}
                    ).to_df()
    
    # 检索多媒体表中，是否有属于该chunk的
    chunk_uuids = chunk_res["chunk_uuid"].tolist()
    # 图片表
    table_image = db_object.get_table("table_image")
    image_res = table_image.output(["image_uuid", "chunk_uuid", "images_placeholder", "image_path"]
                    ).filter(f"chunk_uuid in {tuple(chunk_uuids)}"
                    ).to_df()
    # 表格表
    table_table = db_object.get_table("table_table")
    table_res = table_table.output(["table_uuid", "chunk_uuid", "chunk_tables_placeholder", "table_content"]
                    ).filter(f"chunk_uuid in {tuple(chunk_uuids)}"
                    ).to_df()
    # 公式表
    table_formula = db_object.get_table("table_formula")
    formula_res = table_formula.output(["formula_uuid", "chunk_uuid", "chunk_formulas_placeholder", "formula_content"]
                    ).filter(f"chunk_uuid in {tuple(chunk_uuids)}"
                    ).to_df()
    
    # 还原所有chunk 到字段 restored_content。按章节和内部顺序组织
    restored_chunk_res = restore_all_chunks(chunk_res, image_res, table_res, formula_res)
    
    return restored_chunk_res


# 重排
def rerank(df, query, cloumn, model, threshold=0.0):
    """
    输入：检索得到的df，查询，待重排的字段，重排模型，阈值
    输出：重排筛选后的df
    """
    chunks = df[cloumn].tolist()
    # 组织成对结构
    sentence_pairs = [[query, chunk] for chunk in chunks]
    # 计算
    scores = model.compute_score(sentence_pairs, max_length=1024)
    # 添加到df上
    df = df.copy()
    df['rerank_score'] = scores
    # 过滤，大于等于阈值才行
    filtered_df = df[df['rerank_score'] >= threshold].reset_index(drop=True)
    # 按照重排分数降序排序
    sorted_df = filtered_df.sort_values(by='rerank_score', ascending=False).reset_index(drop=True)
    
    return sorted_df


# 为了给回复加引用，给内容的加上对应的论文数据
def chunk_filter_paper(db_object, paper_uuids):

    table_paper = db_object.get_table("table_paper")
    res = table_paper.output(["paper_uuid", "title", "author", "file_path"]
            ).filter(f"paper_uuid in {tuple(paper_uuids)}"
            ).to_df()
    
    return res

# 加引用
def add_citations(answer, chunk_contents, references, rerank_model, threshold=0.85):
    """
    对回答的每个句子计算与检索到的文献片段的相似度，添加引用。
    
    参数：
        answer (str): 生成的回答
        chunk_contents (list): 检索到的chunk内容
        references(list)：检索到的chunk对应的论文信息
        rerank_model: 用于相似度计算的模型
        threshold (float): 相似度阈值，默认0.7。只有大于等于的才能是引用
        
    返回：
        final_answer (str): 添加引用后的最终回答
        out_references(list): 回答下面的参考文献
    """
    # 将回答进行分句
    sentences = split_sentences(answer, "answer")
    # 引用映射：引用内容 -> 引用编号
    citation_map = {}
    current_id = 1
    final_sentences = [] # 回复内添加引用
    out_references = [] # 回复外的参考文献+关键句(使用句子和对应的chunk里拆一个最关键的句子)

    # 对每个句子计算与chunk的相似度
    for sentence in sentences:
        max_similarity = 0 # 最大的相似度，考虑后续有更相似的参考
        best_ref = None # 每个句子最相关的chunk内容的paper信息
        best_chunk = "" # 每个句子最相关的chunk内容

        for ref, chunk in zip(references, chunk_contents):
            similarity = compute_similarity(sentence, chunk, rerank_model)
            if similarity > max_similarity:
                max_similarity = similarity
                best_chunk = chunk
                best_ref = ref
        # 如果最大相似度超过阈值，添加引用
        if max_similarity >= threshold:
            # 检查引用是否已经存在，使用关键句子级别
            # 回复外部的参考文献
            key_ref_sen = max_similarity_ref(sentence, best_chunk, rerank_model) # 每个chunk的关键句子
            if key_ref_sen not in citation_map: # 仅在没有的时候才更新id
                citation_map[key_ref_sen] = current_id # 字典：{key_sen: id}
                ref_id = current_id
                current_id += 1
                # 添加引用编号到句子末尾
                sentence += f"[{ref_id}]"
                # 添加到参考文献列表
                out_references.append(f"[{ref_id}] {best_ref} - \"{key_ref_sen}\"")
            else:
                # 已存在的关键句子，使用现有的引用编号
                ref_id = citation_map[key_ref_sen]
                sentence += f"[{ref_id}]"

        final_sentences.append(sentence)
        
    # 合并句子，生成最终回答
    final_answer = ''.join(final_sentences)
    # out_references = '\n'.join(out_references) # 下面的参考文献
    
    return final_answer, out_references

# 中文分句
def split_sentences(text, flag):
    """
    将文本按照中文句子分割符进行分割。
    
    参数：
        text (str): 输入文本
        flag(str): 标识是处理回答还是参考文本
        
    返回：
        sentences (List[str]): 分割后的句子列表
    """
    import re
    # 使用正则表达式按照中文句子结束符进行分割。区别在于冒号
    if flag == "answer":
        sentences = re.split('(。|！|\!|\.|？|\?|：)', text)
    else:
        sentences = re.split('(。|！|\!|\.|？|\?)', text)
    # 合并标点符号
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    return sentences
# 计算两个句子的相似度，重排
def compute_similarity(sentence1, sentence2, rerank_model):
    """
    计算两个句子之间的相似度。
    
    参数：
        sentence1 (str): 句子1
        sentence2 (str): 句子2
        rerank_model: 重排模型
        
    返回：
        similarity (float): 相似度得分。0到1之间，不确定
    """
    sentence_pairs = [sentence1, sentence2]
    score = rerank_model.compute_score(sentence_pairs, max_length=1024)

    return score
# 在得到参考文献后，需要定位到具体的chunk内容，就使用query和这个chunk里的进行相似度计算
def max_similarity_ref(sentence, chunk_content, rerank_model):
    
    # chunk内容分句
    sentences = split_sentences(chunk_content, "chunk")
    sentence_pairs = [[sentence, sen] for sen in sentences]
    scores = rerank_model.compute_score(sentence_pairs, max_length=1024)
    # 找到得分最高的句子索引
    top = np.argmax(scores)

    return sentences[top]


# 使用文本检索图片，只返回最相关的第一个图
def text_to_image(db_object, query, model):

    table = db_object.get_table("table_image")
    content_embed = model.encode_text(query, task='retrieval.query')
    image_res = table.output(["image_uuid", "chunk_uuid", "images_placeholder", "image_path"]
                    ).match_dense("image_embed", content_embed, "float", "cosine", 3
                    ).to_df()
    return image_res

# 过滤，根据uuid
def filter_chunk(table_name, uuids=()):
    
    table = db_object.get_table(table_name)
    
    res = table.output(["*"]).filter(f"uuid in {tuple(uuids)}").match_text("chunk_content", "网络", 2).fusion("rrf", 2).to_pl()
    
    return res

# 全文检索
def query_fulltext(table_name, column, query, k=3):
    """
    输入：数据表，待检索的字段，检索词,top_k
    输出：匹配到的df
    """
    table_text = db_object.get_table(table_name)
    
    res = table_text.output(["*", "_score"]).match_text(column, query, k).to_df() 
    
    return res

# 纯向量检索，文本
def search_embed_text(model, column="chunk_embed", table_name="table_text", query="你好", k=3, threshold=0.0):
    
    table_text = db_object.get_table(table_name)

    query_embed = model.encode_text(query, task='retrieval.query') # [0.1,0.1...]

    res = table_text.output(["uuid","file_path","chunk_content", "_similarity"]
                            ).match_dense(
                            column, query_embed, "float", "cosine", k,
                            {"threshold": str(threshold)}
                            ).to_df()

    return res


# 纯向量检索，图片.以文搜图
def search_image(model, table_name="table_image", query="bs64", k=3, threshold=0.0):
    
    table_text = db_object.get_table(table_name)
    
    # query_embed = model.encode_image(query) # [0.1,0.1...]
    query_embed = model.encode_text(query, task='retrieval.query')

    res = table_text.output(["uuid","image_path","image_content", "_similarity"]
                            ).match_dense(
                            "image_embed", query_embed, "float", "cosine", k,
                            {"threshold": str(threshold)}
                            ).to_pl()
    
    return res

# 混合检索
def hybid_search(db_object, model, dense_column="chunk_embed", fulltext_column="chunk_content", table_name="table_text", query="你好", k=3, threshold=0.0):
    """
    输入：嵌入的字段，文本字段，检索词
    输出：混合检索后的df
    """
    table_text = db_object.get_table(table_name)
    query_embed = model.encode_text(query, task='retrieval.query') # [0.1,0.1...]

    res = table_text.output(["*", "_score"]).match_dense(
            dense_column, query_embed, "float", "cosine", k,{"threshold": str(threshold)}
            ).match_text(fulltext_column, query, k).fusion("rrf",3).to_df()

    return res


def find_image(db_object, uuids):
    # 根据重排后的df的uuid找到图片
    table_image = db_object.get_table("table_image")
    res = table_image.output(["uuid","image_path","image_content"]).filter(f"uuid in {tuple(uuids)}").to_df()
    
    return res


def ner(text):
    keywords = jieba.analyse.extract_tags(text, topK=2, withWeight=False, allowPOS=())

    return keywords




if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 向量模型初始化
    model = AutoModel.from_pretrained('autodl-tmp/models/jina-clip-v2', trust_remote_code=True)
    model = model.to(device)
    model.eval()
    # 重排模型初始化
    rerank_model = AutoModelForSequenceClassification.from_pretrained('autodl-tmp/models/jina-reranker-v2-base-multilingual',trust_remote_code=True)
    rerank_model = rerank_model.to(device)
    rerank_model.eval()

    # 连接数据库服务端，本地文件夹，必须是绝对路径
    infinity_object = infinity.connect("/root/RAG/database", config_path="/root/RAG/infinity_config/infinity_conf.toml")
    db_object = infinity_object.get_database("paper")

    # 查询测试
    query = "超网络模型"

    res = text_to_image(db_object, query, model)
    print(res)
    h=1