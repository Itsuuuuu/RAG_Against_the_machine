from pathlib import Path
from chonkie import Chunk, CodeChunker, RecursiveChunker


def collect_py_files(path: Path) -> list[Path]:
	py_file_list = []
	for file in path.rglob("*.py"):
		py_file_list.append(file)
	return py_file_list


def collect_md_files(path: Path) -> list[Path]:
	# Les .txt sont du texte brut : meme strategie de decoupage que le
	# Markdown. 3 % des sources docs de reference sont dans des .txt.
	md_file_list = []
	for pattern in ("*.md", "*.txt"):
		for file in path.rglob(pattern):
			md_file_list.append(file)
	return md_file_list


def read_text(path: Path) -> str:
	# Encodage explicite : l'encodage par defaut depend de la machine, et un
	# fichier mal encode ne doit pas faire planter toute l'indexation.
	with open(path, "r", encoding="utf-8", errors="replace") as file:
		return file.read()


def py_file_chunker(path: Path, chunker: CodeChunker) -> list[Chunk]:
	data = read_text(path)
	chunks = chunker.chunk(data)
	return chunks


def md_file_chunker(path: Path, chunker: RecursiveChunker) -> list[Chunk]:
	data = read_text(path)
	chunks = chunker.chunk(data)
	return chunks
