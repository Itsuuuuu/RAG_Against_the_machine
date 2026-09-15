import time
import uuid
from pathlib import Path

from rank_bm25 import BM25Okapi

from .generator import Questions
from .indexer import build_bm25_index, build_index, load_index, save_index
from .metrics import (
	MAX_CONTEXT_LENGTH,
	count_oversized_sources,
	evaluate_results,
)
from .models import ChunkData, MinimalSearchResults, StudentSearchResults
from .retriever import (
	load_questions,
	load_search_results,
	search,
	search_dataset,
)

# Valeurs par defaut : tous les chemins restent configurables en argument.
DEFAULT_RAW_DIRECTORY = "data/raw"
DEFAULT_INDEX_PATH = "data/processed/index.pkl"
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_K = 5

PREVIEW_LENGTH = 110


def preview(text: str, length: int = PREVIEW_LENGTH) -> str:
	# Une seule ligne, espaces normalises, coupee proprement.
	flat = " ".join(text.split())
	if len(flat) <= length:
		return flat
	return flat[:length].rstrip() + "..."


def print_sources(query: str, sources: list[ChunkData]) -> None:
	print(f'\nTop {len(sources)} sources for: "{query}"\n')
	if not sources:
		print("  (no source found)")
		return
	for rank, source in enumerate(sources, start=1):
		span = f"{source.first_character_index}-{source.last_character_index}"
		print(f"{rank:3d}. {source.file_path}  [{span}]")
		print(f"     {preview(source.text)}")


def print_answer(answer: str) -> None:
	print("\nAnswer:")
	for line in answer.splitlines() or [""]:
		print(f"  {line}")
	print()


def output_path_for(input_path: Path, save_directory: Path) -> Path:
	# Le fichier de sortie porte le nom du fichier d'entree, dans le dossier
	# demande (cree si besoin) : c'est ce que les scripts d'examen attendent.
	save_directory.mkdir(parents=True, exist_ok=True)
	return save_directory / input_path.name


class CLI:
	"""RAG against the machine: index a codebase, search it, answer questions."""

	# Rien de couteux ici : Fire instancie CLI a chaque commande, y compris
	# pour un simple `search`. Chaque commande charge ce dont elle a besoin.

	def _load_index(self, index_path: str) -> tuple[BM25Okapi, list[ChunkData]]:
		path = Path(index_path)
		if not path.is_file():
			raise FileNotFoundError(
				f"index not found at {path}: run `index` first"
			)
		return load_index(path)

	def _load_generator(self) -> Questions:
		generator = Questions()
		generator.load_model()
		return generator

	def index(
		self,
		max_chunk_size: int = DEFAULT_CHUNK_SIZE,
		raw_directory: str = DEFAULT_RAW_DIRECTORY,
		index_path: str = DEFAULT_INDEX_PATH,
	) -> None:
		"""Ingest raw_directory and persist the BM25 index under index_path."""
		if max_chunk_size <= 0 or max_chunk_size > MAX_CONTEXT_LENGTH:
			print(f"max_chunk_size must be in 1..{MAX_CONTEXT_LENGTH}")
			return
		root = Path(raw_directory)
		if not root.is_dir():
			print(f"raw directory not found: {root}")
			return
		start = time.time()
		chunks = build_index(root, max_chunk_size)
		if not chunks:
			print(f"no .py or .md file found under {root}")
			return
		bm25, chunks = build_bm25_index(chunks)
		save_path = Path(index_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		save_index(bm25, chunks, save_path)
		print(
			f"Ingestion complete! {len(chunks)} chunks indexed in "
			f"{time.time() - start:.1f}s, saved under {save_path.parent}/"
		)

	def search(
		self,
		query: str,
		k: int = DEFAULT_K,
		index_path: str = DEFAULT_INDEX_PATH,
	) -> None:
		"""Print the top-k sources for a single query."""
		query = str(query)
		try:
			bm25, chunks = self._load_index(index_path)
		except (OSError, ValueError) as error:
			print(f"cannot load index: {error}")
			return
		sources = search(query, bm25, chunks, k)
		print_sources(query, sources)

	def search_dataset(
		self,
		dataset_path: str,
		k: int = DEFAULT_K,
		save_directory: str = "data/output/search_results",
		index_path: str = DEFAULT_INDEX_PATH,
	) -> None:
		"""Search every question of a dataset, write a StudentSearchResults JSON."""
		try:
			bm25, chunks = self._load_index(index_path)
			dataset = load_questions(Path(dataset_path))
		except (OSError, ValueError) as error:
			print(f"cannot run search_dataset: {error}")
			return
		results = search_dataset(dataset, bm25, chunks, k)
		oversized = count_oversized_sources(results)
		if oversized:
			print(
				f"warning: {oversized} sources exceed {MAX_CONTEXT_LENGTH} "
				"characters, the moulinette will reject this file"
			)
		output = output_path_for(Path(dataset_path), Path(save_directory))
		with open(output, "w", encoding="utf-8") as file:
			file.write(results.model_dump_json(indent=2))
		print(f"Saved student_search_results to {output}")

	def answer(
		self,
		query: str,
		k: int = DEFAULT_K,
		index_path: str = DEFAULT_INDEX_PATH,
	) -> None:
		"""Answer a single query from the top-k retrieved sources."""
		query = str(query)
		try:
			bm25, chunks = self._load_index(index_path)
		except (OSError, ValueError) as error:
			print(f"cannot load index: {error}")
			return
		sources = search(query, bm25, chunks, k)
		print_sources(query, sources)
		if not sources:
			print("\nNo source retrieved: nothing to ground an answer on.\n")
			return
		result = MinimalSearchResults(
			question_id=str(uuid.uuid4()),
			question=query,
			retrieved_sources=sources,
		)
		results = StudentSearchResults(search_results=[result], k=k)
		generator = self._load_generator()
		answered = generator.answer_questions(results, chunks)
		print_answer(answered.search_results[0].answer)

	def answer_dataset(
		self,
		student_search_results_path: str,
		save_directory: str = "data/output/search_results_and_answer",
		index_path: str = DEFAULT_INDEX_PATH,
	) -> None:
		"""Generate answers for a StudentSearchResults JSON, write the result."""
		try:
			results = load_search_results(Path(student_search_results_path))
			_, chunks = self._load_index(index_path)
		except (OSError, ValueError) as error:
			print(f"cannot run answer_dataset: {error}")
			return
		print(f"Loaded {len(results.search_results)} questions")
		generator = self._load_generator()
		answered = generator.answer_questions(results, chunks)
		output = output_path_for(
			Path(student_search_results_path), Path(save_directory)
		)
		with open(output, "w", encoding="utf-8") as file:
			file.write(answered.model_dump_json(indent=2))
		print(f"Saved student_search_results_and_answer to {output}")

	def evaluate(
		self,
		student_search_results_path: str,
		dataset_path: str,
	) -> None:
		"""Report recall@k of a StudentSearchResults file against ground truth."""
		try:
			results = load_search_results(Path(student_search_results_path))
			dataset = load_questions(Path(dataset_path))
		except (OSError, ValueError) as error:
			print(f"cannot evaluate: {error}")
			return
		oversized = count_oversized_sources(results)
		print(f"Student data is valid: {oversized == 0}")
		if oversized:
			print(f"  ({oversized} sources exceed {MAX_CONTEXT_LENGTH} characters)")
		print("Evaluation Results")
		print("=" * 40)
		recalls = evaluate_results(results, dataset)
		print("  ".join(f"Recall@{k}: {value:.3f}" for k, value in recalls.items()))
