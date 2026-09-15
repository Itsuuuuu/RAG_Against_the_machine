from pathlib import Path
from .models import (
	RagDataset,
	StudentSearchResults,
	ChunkData,
	MinimalSearchResults,
)
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from .indexer import clean_text, STOPWORDS, remove_stopwords


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
	tokens = remove_stopwords(clean_text(question).split(" "), STOPWORDS)
	# "".split(" ") donne [""] : on ecarte les tokens vides.
	return [token for token in tokens if token]


def search(question: str, bm25: BM25Okapi, chunks: list[ChunkData], k: int) -> list[ChunkData]:
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


def search_dataset(dataset: RagDataset, bm25: BM25Okapi, chunks: list[ChunkData], k: int) -> StudentSearchResults:
	result = []
	for question in tqdm(dataset.rag_questions, desc="searching"):
		sources = search(question.question, bm25, chunks, k)
		minimal_result = MinimalSearchResults(
			question_id=question.question_id,
			question=question.question,
			retrieved_sources=sources,
		)
		result.append(minimal_result)
	return StudentSearchResults(search_results=result, k=k)
