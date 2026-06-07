import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

class DualLayerRAG:
    def __init__(self):
        print("🔧 初始化 RAG 系統...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # 改成持久化，第一次建立後之後直接讀取不重建
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self._build_knowledge_base()
        print("✅ RAG 系統就緒")

    def _load_data(self):
        """直接載入 data 目錄下的 JSON 檔案"""
        
        mitre_path = "data/mitre_official.json"
        fewshot_path = "data/fewshot_real.json"

        if not os.path.exists(mitre_path):
            raise FileNotFoundError(f"找不到 {mitre_path}，請先建立或下載該檔案。")
        with open(mitre_path, "r", encoding="utf-8") as f:
            mitre_data = json.load(f)
        print("  📂 使用 MITRE 官方資料")

        if not os.path.exists(fewshot_path):
            raise FileNotFoundError(f"找不到 {fewshot_path}，請先建立或下載該檔案。")
        with open(fewshot_path, "r", encoding="utf-8") as f:
            fewshot_data = json.load(f)
        print("  📂 使用真實釣魚郵件資料集")

        return mitre_data, fewshot_data

    def _build_knowledge_base(self):
        mitre_data, fewshot_data = self._load_data()
        
        # 第一層：MITRE 向量化
        self.mitre_collection = self.client.get_or_create_collection("mitre_kb")
        for item in mitre_data:
            text = f"{item['name']}: {item['description']}. 特徵：{item['indicators']}"
            embedding = self.model.encode(text).tolist()
            self.mitre_collection.upsert(
                ids=[item['id']],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"technique_id": item['id'], "name": item['name']}]
            )

        # 第二層：Few-Shot 向量化
        self.fewshot_collection = self.client.get_or_create_collection("fewshot_examples")
        for i, example in enumerate(fewshot_data):
            embedding = self.model.encode(example['email']).tolist()
            self.fewshot_collection.upsert(
                ids=[f"example_{i}"],
                embeddings=[embedding],
                documents=[example['email']],
                metadatas=[{
                    "label": example['label'],
                    "technique": example['technique'],
                    "reason": example['reason']
                }]
            )

    def retrieve_mitre_context(self, email_body: str, top_k: int = 2) -> list:
        embedding = self.model.encode(email_body).tolist()
        results = self.mitre_collection.query(query_embeddings=[embedding], n_results=top_k)
        return [
            {"technique": results['metadatas'][0][i]['technique_id'],
             "description": results['documents'][0][i]}
            for i in range(len(results['documents'][0]))
        ]

    def retrieve_fewshot_examples(self, email_body: str, top_k: int = 3) -> list:
        embedding = self.model.encode(email_body).tolist()
        results = self.fewshot_collection.query(query_embeddings=[embedding], n_results=top_k)
        return [
            {"email": results['documents'][0][i],
             "label": results['metadatas'][0][i]['label'],
             "reason": results['metadatas'][0][i]['reason']}
            for i in range(len(results['documents'][0]))
        ]