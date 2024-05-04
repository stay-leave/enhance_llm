from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from pathlib import Path
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

def evaluate_st(
    dataset,
    model_id,
    name,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name
    )
    model = SentenceTransformer(model_id)
    output_path = "project_2/results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)



# 加载本地数据集-验证集
val_dataset = EmbeddingQAFinetuneDataset.from_json("project_2/data/datasets/val_dataset.json")

# 评估
bge = "local:project_2/bge-large-zh-v1.5"
evaluate_st(val_dataset, "project_2/bge-large-zh-v1.5", name="bge")

finetuned_1 = "local:project_2/model_outdir/1.8b-bge"
evaluate_st(val_dataset, "project_2/model_outdir/1.8b-bge", name="1.8b")

finetuned_2 = "local:project_2/model_outdir/7b-bge"
evaluate_st(val_dataset, "project_2/model_outdir/7b-bge", name="7b")