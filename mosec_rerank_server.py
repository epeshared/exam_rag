import mosec
import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
from pprint import pprint

device = "cpu"
use_ipex = "true"
class RerankerWorker(mosec.Worker):
    def __init__(self):
        super().__init__()
        self.device = device
        self.use_ipex = use_ipex
        print(f"Loading BGE-Reranker-Large model... on device={self.device} with use_ipex={self.use_ipex}")
        
        self.tokenizer = AutoTokenizer.from_pretrained('/home/xtang/models/BAAI/bge-reranker-large')
        self.model = AutoModelForSequenceClassification.from_pretrained('/home/xtang/models/BAAI/bge-reranker-large').to(self.device)
        self.model.eval()

        if self.device == "cpu" and self.use_ipex:
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
            vocab_size = self.model.config.vocab_size
            batch_size = 16
            seq_length = 512
            d = torch.randint(vocab_size, size=[batch_size, seq_length]).to(self.device)
            m = torch.randint(1, 2, size=[batch_size, seq_length]).to(self.device)
            # 使用 `torch.jit.trace` 进行优化
            self.model = torch.jit.trace(self.model, [d, m], check_trace=False, strict=False)
            self.model = torch.jit.freeze(self.model)
        
        print("Reranker created.")

    def rerank(self, queries, candidates):
        reranked_results = {}
        for query_id, query_text in queries.items():
            if isinstance(candidates, str):
                print(f"Unexpected candidates format: {candidates}")
                continue
            query_candidates = candidates.get(query_id, [])
            if not query_candidates:
                print(f"No candidates found for query_id: {query_id}")
                continue

            pairs = [[query_text, candidate['text']] for candidate in query_candidates]
            
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                scores = outputs['logits'].view(-1, ).float()  # 从字典中提取logits
                sorted_indexes = scores.argsort(descending=True)
                # 按照 sorted_indexes 对 query_candidates 进行排序
                sorted_candidates = [query_candidates[i] for i in sorted_indexes]
                reranked_results[query_id] = sorted_candidates

        return reranked_results

    def forward(self, data):
        try:
            results = []
            for item in data:
                if isinstance(item, list) and len(item) > 0:
                    item = item[0]  # 如果 item 是列表，获取第一个元素

                if isinstance(item, dict):
                    queries = item.get('queries', {})
                    candidates = item.get('candidates', {})
                    
                    # 调用 rerank 函数，直接传入整个 queries 和 candidates
                    result = self.rerank(queries, candidates)
                    results.append(result)
                else:
                    print(f"Unexpected item format: {type(item)}")
                    continue

            return results
        except Exception as e:
            print("Error in forward method:", str(e))
            raise e

# 启动Mosec服务
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reranker Service")
    parser.add_argument("--rerank-Dev", type=str, default="cpu", help="Specify the device for reranking, e.g., 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument("--use-ipex", type=str, default="True", help="Use IPEX to optimize model when using CPU")

    args, extra = parser.parse_known_args()

    # 打印已解析的参数和未知参数
    print("Known arguments:", args)
    print("extra arguments:", extra)

    # 打印传入的参数
    print("Starting Reranker Service with the following options:")
    pprint(vars(args))

    device=args.rerank_Dev
    use_ipex=args.use_ipex

    service = mosec.Server()
    service.append_worker(RerankerWorker, num=8, max_batch_size=16, max_wait_time=10)
    service.run()
