from chonkie import Chunk, RecursiveChunker, RecursiveRules, CodeChunker
from pathlib import Path
from .models import ChunkData
from .parsing import (
    collect_md_files,
    collect_py_files,
    py_file_chunker,
    md_file_chunker,
    read_text,
)
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import pickle
import string


STOPWORDS = [
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they",
    "how", "what", "why", "when", "where", "which", "who",
    "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "this", "that",
    ]

def create_md_chunker(max_chunk_size: int):
    chunker = RecursiveChunker(
            tokenizer = "character",
            chunk_size = max_chunk_size,
            rules = RecursiveRules(),
            min_characters_per_chunk =24,
        )
    return chunker

def create_py_chunker(max_chunk_size: int):
    chunker = CodeChunker(
            language="python",
            tokenizer="character",
            chunk_size= max_chunk_size,
            include_nodes=False
        )
    return chunker

def byte_to_char_offset(raw: bytes, byte_offset: int) -> int:
    # Nombre de caracteres contenus dans les `byte_offset` premiers octets.
    return len(raw[:byte_offset].decode("utf-8", errors="ignore"))


def chunk_to_data(chunk: Chunk, path: Path, raw: bytes | None = None) -> ChunkData:
    start = chunk.start_index
    end = chunk.end_index
    if raw is not None:
        # CodeChunker s'appuie sur tree-sitter, qui compte en OCTETS ;
        # RecursiveChunker compte en caracteres. On ramene tout en
        # caracteres : sinon un chunk de 2000 caracteres avec des accents
        # depasse 2000 "octets" et la moulinette le rejette, et
        # text[start:end] ne retombe pas sur le bon texte.
        start = byte_to_char_offset(raw, start)
        end = byte_to_char_offset(raw, end)
    return ChunkData(
        file_path= str(path),
        first_character_index= start,
        last_character_index= end,
        text= chunk.text,
    )


def split_oversized(chunk: ChunkData, max_chunk_size: int) -> list[ChunkData]:
    # chonkie ne garantit pas strictement chunk_size (un groupe de noeuds
    # de l'AST peut deborder de quelques caracteres). Or la moulinette
    # rejette TOUT le fichier de resultats si une seule source depasse
    # 2000 caracteres. On redecoupe donc mecaniquement les rares chunks
    # trop longs, en conservant des offsets exacts.
    if len(chunk.text) <= max_chunk_size:
        return [chunk]
    pieces = []
    for offset in range(0, len(chunk.text), max_chunk_size):
        piece = chunk.text[offset:offset + max_chunk_size]
        start = chunk.first_character_index + offset
        pieces.append(ChunkData(
            file_path=chunk.file_path,
            first_character_index=start,
            last_character_index=start + len(piece),
            text=piece,
        ))
    return pieces


def build_index(root: Path, max_chunk_size: int = 2000) -> list[ChunkData]:
    py_files = collect_py_files(root)
    md_files = collect_md_files(root)
    py_chunker = create_py_chunker(max_chunk_size)
    md_chunker = create_md_chunker(max_chunk_size)
    list_py = []
    list_md = []

    for file in tqdm(py_files, desc="python files"):
        chunks = py_file_chunker(file, py_chunker)
        raw = read_text(file).encode("utf-8")
        for chunk in chunks:
            data = chunk_to_data(chunk, file, raw)
            list_py.extend(split_oversized(data, max_chunk_size))

    for file in tqdm(md_files, desc="markdown files"):
        chunks = md_file_chunker(file, md_chunker)
        for chunk in chunks:
            data = chunk_to_data(chunk, file)
            list_md.extend(split_oversized(data, max_chunk_size))
    return(list_py + list_md)

def tokenize_corpus(chunks: list[ChunkData]) -> list[list[str]]:
    result = []
    for chunk in chunks:
        result.append(remove_stopwords(clean_text(chunk.text).split(" "), STOPWORDS))
    return result

def clean_text(text: str) -> str:
    str_lower = text.lower()
    table = str.maketrans("", "", string.punctuation)
    cleaned_text = str_lower.translate(table)
    return cleaned_text

def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str]:
    result = []
    for token in tokens:
        if token not in stopwords:
            result.append(token)
    return result

def build_bm25_index(chunks: list[ChunkData]) -> tuple[BM25Okapi, list[ChunkData]]:
    tokenized = tokenize_corpus(chunks)
    bm25 = BM25Okapi(tokenized)
    return (bm25, chunks)

def save_index(bm25: BM25Okapi, chunks: list[ChunkData], path: Path) -> None:
    with open(path, "wb") as file:
        pickle.dump((bm25, chunks), file)

def load_index(path: Path) -> tuple[BM25Okapi, list[ChunkData]]:
    with open(path, "rb") as file:
        data = pickle.load(file)
    return (data)
