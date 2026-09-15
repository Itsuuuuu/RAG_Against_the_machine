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
import re


STOPWORDS = frozenset([
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they",
    "how", "what", "why", "when", "where", "which", "who",
    "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "this", "that",
    ])

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

# Un "mot" : lettres, chiffres et underscore. Tout le reste (ponctuation,
# parentheses, points, retours a la ligne) est un separateur. C'est le point
# cle pour le code : `get_activation_formats(self)` doit donner le token
# `get_activation_formats`, pas `getactivationformatsself`.
WORD = re.compile(r"[A-Za-z0-9_]+")

# Morceaux d'un identifiant camelCase : "FusedMoEActivationFormat" ->
# ["Fused", "Mo", "E", "Activation", "Format"].
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Parametres BM25. b regle la normalisation par la longueur du document :
# les chunks de code ont des longueurs tres variables, une normalisation
# moderee (0.5 au lieu de 0.75) donne +2 a +4 pts de recall@5 sur le code
# sans rien couter aux docs. k1 n'est pas discriminant sur nos datasets.
BM25_K1 = 1.2
BM25_B = 0.5


def split_identifier(word: str) -> list[str]:
    # snake_case puis camelCase : "fused_batched_moe" -> [fused, batched, moe]
    parts = []
    for piece in word.split("_"):
        parts.extend(CAMEL.findall(piece))
    return [part.lower() for part in parts if part]


def tokenize(text: str) -> list[str]:
    # Une seule fonction pour le corpus ET les questions : les deux doivent
    # etre decoupes exactement de la meme facon pour que les tokens matchent.
    # Chaque identifiant est indexe entier (pour les questions qui le citent
    # verbatim) ET par sous-mots (pour celles qui le paraphrasent :
    # "activation formats" vs `activation_formats`).
    tokens = []
    for word in WORD.findall(text):
        lower = word.lower()
        if lower not in STOPWORDS:
            tokens.append(lower)
        parts = split_identifier(word)
        if len(parts) > 1:
            tokens.extend(part for part in parts if part not in STOPWORDS)
    return tokens


def tokenize_corpus(chunks: list[ChunkData]) -> list[list[str]]:
    result = []
    for chunk in tqdm(chunks, desc="tokenizing"):
        result.append(tokenize(chunk.text))
    return result


def build_bm25_index(chunks: list[ChunkData]) -> tuple[BM25Okapi, list[ChunkData]]:
    tokenized = tokenize_corpus(chunks)
    # Un chunk sans aucun token (ponctuation seule, blancs) ne peut jamais
    # etre retrouve, et sa longueur nulle fait diviser BM25 par zero.
    kept = [(tokens, chunk) for tokens, chunk in zip(tokenized, chunks) if tokens]
    tokenized = [tokens for tokens, _ in kept]
    chunks = [chunk for _, chunk in kept]
    bm25 = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)
    return (bm25, chunks)

def save_index(bm25: BM25Okapi, chunks: list[ChunkData], path: Path) -> None:
    with open(path, "wb") as file:
        pickle.dump((bm25, chunks), file)

def load_index(path: Path) -> tuple[BM25Okapi, list[ChunkData]]:
    with open(path, "rb") as file:
        data = pickle.load(file)
    return (data)
