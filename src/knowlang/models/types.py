from enum import Enum
from typing import List

EmbeddingVector = List[float]
class EmbeddingInputType(Enum):
    DOCUMENT = "document"
    QUERY = "query"