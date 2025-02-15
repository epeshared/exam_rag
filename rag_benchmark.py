import time
import csv
import json
import os
import argparse
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm
import concurrent.futures
import httpx
import itertools
import intel_extension_for_pytorch as ipex
import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pprint import pprint  # 导入 pprint 模块
import transformers 

# 存储步骤执行时间的列表
execution_times = []
tps_list = []
embedding_times = []
search_times = []
rerank_times = []

query_results = {}
rerank_results_dict = {}  # 用于存储所有查询的rerank结果

# 初始化一个全局变量用于累加所有查询的执行时间
total_query_time = 0

# 记录时间的函数
def log_time(step_name, start_time, end_time):
    execution_time = end_time - start_time
    execution_times.append([step_name, f"{execution_time:.3f}"]) 
    print(f"Time taken for {step_name}: {execution_time:.2f} seconds")

# 记录TPS的函数
def log_tps(query_count, total_time):
    tps = query_count / total_time
    tps_list.append(["TPS", f"{tps:.3f}"])
    print(f"TPS (Queries per second): {tps:.2f}")

class EmbeddingGenerator:
    def __init__(self, model_name, device="cpu", use_ipex="True"):
        self.device = device
        self.use_ipex = use_ipex

        if use_ipex == "True" and device == "cpu":
            tmp_model = SentenceTransformer(model_name, device=device)
            self.dim = tmp_model.get_sentence_embedding_dimension() 

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = transformers.AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)

            self.model.eval()
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
            vocab_size = self.model.config.vocab_size
            batch_size = 16
            seq_length = 512
            d = torch.randint(vocab_size, size=[batch_size, seq_length])
            t = torch.randint(0, 1, size=[batch_size, seq_length])
            m = torch.randint(1, 2, size=[batch_size, seq_length])
            model_inputs = [d]
            if "token_type_ids" in self.tokenizer.model_input_names:
                model_inputs.append(t)
            if "attention_mask" in self.tokenizer.model_input_names:
                model_inputs.append(m)
            self.model = torch.jit.trace(self.model, model_inputs, check_trace=False, strict=False)
            self.model = torch.jit.freeze(self.model)
            self.model(*model_inputs)
        else:
            self.model = SentenceTransformer(model_name, device=device)
            self.dim = self.model.get_sentence_embedding_dimension()      
        
    def generate_embeddings(self, texts, batch_size=32, show_progress_bar=True):
        embeddings = []

        def mean_pooling(model_output, attention_mask):
            # First element of model_output contains all token embeddings
            token_embeddings = model_output["last_hidden_state"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        total_texts = len(texts)
        
        if self.use_ipex == "True" and self.device == "cpu":
            with tqdm(total=total_texts, desc="Generating embeddings", unit="text", disable=not show_progress_bar) as pbar:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    # Tokenize sentences
                    encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                    inputs = encoded_input.to(self.device)
                    # Compute token embeddings
                    model_output = self.model(**inputs)
                    # Perform pooling
                    batch_embeddings = mean_pooling(model_output, inputs["attention_mask"])
                    # Normalize embeddings
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                    embeddings.extend(batch_embeddings.cpu().numpy())
                    pbar.update(len(batch_texts))  # Update progress by the number of processed texts
                    # print(f"--->finish {i} text")
        else:
            with tqdm(total=total_texts, desc="Generating embeddings", unit="text", disable=not show_progress_bar) as pbar:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
                    embeddings.extend(batch_embeddings)
                    pbar.update(len(batch_texts))  # Update progress by the number of processed texts

        return embeddings

  
    
    def get_model_dim(self):
        return self.dim

class MilvusHandler:
    def __init__(self, collection_name, embedding_dim, force_del=False):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        connections.connect()
        if utility.has_collection(self.collection_name):
            if force_del:
                print(f"Collection {self.collection_name} already exists, deleting...")
                utility.drop_collection(self.collection_name)
                print(f"Collection {self.collection_name} deleted.")
            else:
                print(f"Collection {self.collection_name} already exists.")
        else:
            print(f"Collection {self.collection_name} does not exist, creating...")
        
        if not utility.has_collection(self.collection_name):
            self.create_collection()
        
        self.collection = Collection(self.collection_name)

    def create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        schema = CollectionSchema(fields, "Document embeddings for retrieval")
        collection = Collection(self.collection_name, schema)
        collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 100}})
        collection.load()

    def insert_embeddings(self, doc_ids, embeddings, batch_size=100):
        for i in range(0, len(doc_ids), batch_size):
            batch_doc_ids = doc_ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            entities = [batch_doc_ids, batch_embeddings]
            self.collection.insert(entities)
        self.collection.flush()
        self.collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 100}})
        self.collection.load()

    def search(self, query_embedding, top_k=10):
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        result = self.collection.search([query_embedding], "embedding", param=search_params, limit=top_k, output_fields=["id"])
        return {hit.id: hit.distance for hit in result[0]}

def generate_embeddings_for_corpus(embedder, corpus, batch_size=32, num_workers=3):
    doc_ids = list(corpus.keys())
    doc_texts = [doc['text'] for doc in corpus.values()]

    def generate_embeddings_batch(batch_texts):
        return embedder.generate_embeddings(batch_texts, batch_size=batch_size, show_progress_bar=False)

    total_batches = len(doc_texts) // batch_size + (1 if len(doc_texts) % batch_size != 0 else 0)
    corpus_embeddings = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_embeddings_batch, doc_texts[i:i + batch_size])
            for i in range(0, len(doc_texts), batch_size)
        ]

        with tqdm(total=total_batches, desc="Generating embeddings", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches | {rate_fmt} batches/s | {elapsed}s elapsed") as pbar:
            for future in concurrent.futures.as_completed(futures):
                corpus_embeddings.extend(future.result())
                pbar.update(1)
    
    end_time = time.time()
    log_time("Generate embeddings", start_time, end_time)
    
    return doc_ids, corpus_embeddings

def insert_embeddings_to_milvus(retriever, doc_ids, corpus_embeddings, batch_size=100):
    retriever.insert_embeddings(doc_ids, corpus_embeddings, batch_size=batch_size)

# 加载数据
def load_data():
    print("Loading Data...")
    dataset = "nq"
    dataset_dir = os.path.join("datasets", dataset)
    
    if (os.path.exists(dataset_dir) and os.path.isdir(dataset_dir)):
        print(f"Dataset '{dataset}' already exists, skipping download.")
        data_path = dataset_dir
    else:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, "datasets")
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels    

# 从文件中读取端口号列表
def load_ports_from_file(filepath):
    with open(filepath, 'r') as file:
        ports = [int(line.strip()) for line in file if line.strip()]
    return ports

# 预处理数据
def preprocess_data(corpus, queries, qrels, max_corpus_size=1000, max_queries_size=100):
    print("process max_corpus_size:", max_corpus_size)
    print("process max_queries_size:", max_queries_size)
    corpus = {k: corpus[k] for k in list(corpus)[:max_corpus_size]}
    queries = {k: queries[k] for k in list(queries)[:max_queries_size]}
    qrels = {k: qrels[k] for k in list(queries) if k in qrels}
    return corpus, queries, qrels

def rerank_using_mosec(data_for_reranking, port):
    try:
        url = f"http://127.0.0.1:{port}/inference"
        with httpx.Client() as client:
            response = client.post(url, json=[data_for_reranking], timeout=300000)
            return response.json()  # 获取第一个（也是唯一的）结果
    except Exception as e:
        print(f"Reranking failed on port {port}: {e}")
        return None  # 如果 reranking 失败，返回 None 或者原始结果

def process_single_query(query_id, query_text, retriever, embedder, qrels, port_cycle, record_results=False):
    global total_query_time

    # 记录整个函数执行的开始时间
    overall_start_time = time.time()
    
    # 记录生成嵌入的时间
    start_time = time.time()
    query_embedding = embedder.generate_embeddings([query_text], batch_size=1, show_progress_bar=False)[0]
    end_time = time.time()
    embedding_time = end_time - start_time
    embedding_times.append(embedding_time)

    # 记录搜索的时间
    start_time = time.time()
    result = retriever.search(query_embedding)
    end_time = time.time()
    search_time = end_time - start_time
    search_times.append(search_time)

    # 将结果存储在字典中，如果记录结果的选项打开
    if record_results:
        query_results[query_id] = result

    # 将 search_results 转换为 candidates 格式
    candidates = [{"text": get_document_text(doc_id), "score": score} for doc_id, score in result.items()]

    # 准备数据用于 reranking
    data_for_reranking = {
        "queries": {query_id: query_text},
        "candidates": {query_id: candidates}
    }

    # 获取当前轮询的端口号
    port = next(port_cycle)

    start_time = time.time()
    reranked_results = rerank_using_mosec(data_for_reranking, port)
    end_time = time.time()
    rerank_time = end_time - start_time
    rerank_times.append(rerank_time)

    if reranked_results is not None:
        # 将 query_text 也存储在 rerank_results_dict 中
        rerank_results_dict[query_id] = {
            "query_text": query_text,
            "results": reranked_results[query_id]
        }
    else:
        rerank_results_dict[query_id] = result  # 如果 reranking 失败，保存原始搜索结果

    # 记录整个函数执行的结束时间
    overall_end_time = time.time()
    overall_execution_time = overall_end_time - overall_start_time
    
    # 将此查询的执行时间累加到 total_query_time 中
    total_query_time += overall_execution_time

def save_rerank_results_to_json(filename="rerank_results.json"):
    # 对每个查询的 rerank 结果按照分数从高到低进行排序
    sorted_rerank_results_dict = {}
    for query_id, data in rerank_results_dict.items():
        sorted_results = sorted(data["results"], key=lambda x: x['score'], reverse=True)
        sorted_rerank_results_dict[query_id] = {
            "query_text": data["query_text"],
            "results": sorted_results
        }

    # 将排序后的结果保存为 JSON 文件
    with open(filename, 'w') as f:
        json.dump(sorted_rerank_results_dict, f, indent=4)
    print(f"All rerank results saved to {filename}")


def save_query_results_to_json(filename="query_results.json"):
    with open(filename, 'w') as f:
        json.dump(query_results, f, indent=4)
    print(f"Query results saved to {filename}")

def get_document_text(doc_id):
    return corpus.get(doc_id, {}).get('text', '')        
  

# 评估单个结果
def evaluate_single_result(qrel, result):
    evaluator = EvaluateRetrieval(None)
    result_str = {str(doc_id): score for doc_id, score in result.items()}
    ndcg, _map, recall, precision = evaluator.evaluate({list(qrel.keys())[0]: qrel}, {list(qrel.keys())[0]: result_str}, k_values=[1, 3, 5, 10])
    print(f"Evaluation completed. NDCG@10: {ndcg.get('NDCG@10', 'N/A')}, MAP@10: {_map.get('MAP@10', 'N/A')}, Recall@10: {recall.get('Recall@10', 'N/A')}, Precision@10: {precision.get('Precision@10', 'N/A')}")
    return ndcg, _map, recall, precision

# 保存执行时间和TPS到CSV
def save_execution_times_to_csv():
    global total_query_time

    # 计算平均时间
    avg_query_time = total_query_time / len(queries) if len(queries) > 0 else 0

    avg_embedding_time = sum(embedding_times) / len(queries) if queries else 0
    avg_search_time = sum(search_times) / len(queries) if queries else 0
    avg_rerank_time = sum(rerank_times) / len(queries) if queries else 0

    csv_filename = "execution_times.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Execution Time (seconds)"])
        writer.writerows(execution_times)        
        writer.writerow(["Single Query Embedding Generation", f"{avg_embedding_time:.3f}"])
        writer.writerow(["Retrieval Avg. Time", f"{avg_search_time:.3f}"])        
        writer.writerow(["Reranking Avg. Time", f"{avg_rerank_time:.3f}"])
        writer.writerow(["Average Single Query Time", f"{avg_query_time:.3f}"])  # 将平均时间保存到CSV文件中
        writer.writerow(["Query TPS", tps_list[0][1]])  # 将TPS保存到CSV文件中
    print(f"Execution times and TPS saved to {csv_filename}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus Indexing and Retrieval")
    parser.add_argument("--force-del", type=str2bool, nargs='?', const=True, default=False, help="Force delete collection and regenerate embeddings")
    parser.add_argument("--force-emb", type=str2bool, nargs='?', const=True, default=False, help="Force regenerate embeddings and insert into Milvus")
    parser.add_argument("--save-results", type=str2bool, nargs='?', const=True, default=False, help="Save query results to JSON")
    parser.add_argument("--port-file", type=str, default="ports.txt", help="File containing list of Mosec ports")
    parser.add_argument("--embedding-Dev", type=str, default="cpu", help="Specify the device for generating embeddings, e.g., 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument("--use-ipex", type=str, default="True", help="Specify whether to use ipex to accelerate the model")
    args = parser.parse_args()

    # 打印所有命令行输入的参数
    print("Command-line arguments:")
    pprint(vars(args))  # 使用 pprint 打印 args 变量

    # 从文件中加载端口号列表
    ports = load_ports_from_file(args.port_file)
    port_cycle = itertools.cycle(ports)  # 创建循环迭代器

    corpus, queries, qrels = load_data()
    corpus, queries, qrels = preprocess_data(corpus, queries, qrels, len(corpus), len(queries))

    # 实例化 EmbeddingGenerator 和 MilvusHandler
    model_name = "/home/xtang/models/BAAI/bge-large-en-v1.5"
    embedder = EmbeddingGenerator(model_name, args.embedding_Dev, args.use_ipex)
    print("Start Milvus....")
    retriever = MilvusHandler("beir_documents", embedding_dim=embedder.get_model_dim(), force_del=args.force_del)

    if args.force_emb or args.force_del:
        # 生成并插入嵌入
        doc_ids, corpus_embeddings = generate_embeddings_for_corpus(embedder, corpus, batch_size=32)
        insert_embeddings_to_milvus(retriever, doc_ids, corpus_embeddings, batch_size=100)
    else:
        print("Skipping embedding generation and insertion into Milvus.")

    # 使用多线程逐个处理查询
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(process_single_query, query_id, query_text, retriever, embedder, qrels, port_cycle, args.save_results)
            for query_id, query_text in queries.items()
        ]

        # 使用tqdm显示进度条
        with tqdm(total=len(futures), desc="Processing queries", unit="query") as pbar:
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 确保所有查询都被处理
                pbar.update(1)  # 更新进度条

    end_time = time.time()

    # 计算并记录TPS
    total_time = end_time - start_time
    print("total_query_time:", total_query_time)
    print(f"{len(queries)} queries total time: {total_time}")
    log_tps(len(queries), total_time)

    # 保存执行时间和TPS到CSV
    save_execution_times_to_csv()

    # 如果启用了保存查询结果的选项，则保存初步查询结果到JSON文件
    if args.save_results:
        save_query_results_to_json()
        save_rerank_results_to_json()  # 保存rerank结果到JSON文件
