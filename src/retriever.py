from pathlib import Path
from .models import (
    ChunkData,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from .indexer import tokenize


def load_questions(path: Path) -> RagDataset:
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()
    rag_dataset = RagDataset.model_validate_json(data)
    return rag_dataset


def load_search_results(path: Path) -> StudentSearchResults:
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()
    return StudentSearchResults.model_validate_json(data)


def tokenize_query(question: str) -> list[str]:
    # Meme decoupage que le corpus, obligatoirement.
    return tokenize(question)


def to_minimal_source(chunk: ChunkData) -> MinimalSource:
    # Les fichiers de resultats ne portent que la localisation (fichier +
    # intervalle), jamais le texte du chunk.
    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.first_character_index,
        last_character_index=chunk.last_character_index,
    )


def search(
    question: str, bm25: BM25Okapi, chunks: list[ChunkData], k: int
) -> list[ChunkData]:
    tokens = tokenize_query(question)
    if k <= 0 or not tokens:
        return []
    scores = bm25.get_scores(tokens)
    # Aucun token de la requete n'existe dans le corpus (requete absurde) :
    # get_top_n renverrait k chunks arbitraires a score nul. On prefere rien.
    if scores.max() <= 0:
        return []
    best = scores.argsort()[::-1][:k]
    return [chunks[i] for i in best]


def search_dataset(
    dataset: RagDataset, bm25: BM25Okapi, chunks: list[ChunkData], k: int
) -> StudentSearchResults:
    result = []
    for question in tqdm(dataset.rag_questions, desc="searching"):
        sources = search(question.question, bm25, chunks, k)
        minimal_result = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=[to_minimal_source(s) for s in sources],
        )
        result.append(minimal_result)
    return StudentSearchResults(search_results=result, k=k)
