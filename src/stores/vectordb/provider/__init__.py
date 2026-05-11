from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
from typing import List, Dict, Any, Optional


DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclidean": Distance.EUCLID,
}


class QdrantProvider:

    def __init__(self, path: str, distance_metric: str = "cosine"):
        self.client = QdrantClient(path=path)
        self.distance = DISTANCE_MAP.get(distance_metric.lower(), Distance.COSINE)

    def create_collection(self, name: str, dimension: int) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimension, distance=self.distance),
            )

    def upsert(self, collection: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
            for vec, payload in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, query_vector: List[float], k: int = 5, language: Optional[str] = None) -> List[Dict[str, Any]]:
        query_filter = (
            Filter(must=[FieldCondition(key="language", match=MatchValue(value=language))])
            if language else None
        )
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=k,
            query_filter=query_filter,
        )
        return [{"score": hit.score, **hit.payload} for hit in results.points]

    def collection_info(self, name: str) -> Dict[str, Any]:
        existing = [c.name for c in self.client.get_collections().collections]
        if name not in existing:
            return {"exists": False, "points_count": 0, "dimension": None}
        info = self.client.get_collection(name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "dimension": info.config.params.vectors.size,
        }
