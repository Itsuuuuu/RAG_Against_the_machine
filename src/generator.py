import os

# transformers 5 charge les poids avec 4 threads. Sur cette machine, ca
# provoque un deadlock intermittent dans safetensors (PySafeSlice.__getitem__
# -> OnceLock::initialize, vu a la pile d'appel) : le processus reste bloque
# a 0 % CPU pour toujours. Le chargement sequentiel est fiable, et pour un
# modele de 0.6B la difference de temps est negligeable. A positionner
# AVANT l'import de transformers.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

from transformers import (  # noqa: E402
	AutoModelForCausalLM,
	AutoTokenizer,
	PreTrainedTokenizerBase,
	PreTrainedModel
)
from .models import (
	ChunkData,
	MinimalAnswer,
	MinimalSource,
	StudentSearchResults,
	StudentSearchResultsAndAnswer,
)
from tqdm import tqdm
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Nombre max de tokens generes par reponse. Le quickstart Qwen met 32768,
# mais c'est ecrit pour un GPU : sur CPU une seule reponse prendrait des
# minutes. 256 tokens suffisent pour une reponse factuelle.
MAX_NEW_TOKENS = 256

# Consigne donnee au modele : repondre a partir des sources, pas de memoire.
SYSTEM_PROMPT = (
	"You are a technical assistant answering questions about the vLLM "
	"codebase. Answer using only the information contained in the provided "
	"sources. If the sources do not contain the answer, say so plainly "
	"instead of guessing. Be concise and factual."
)


class Questions:
	tokenizer: PreTrainedTokenizerBase
	model: PreTrainedModel

	def load_model(self) -> None:
		# Pour charger le model et tokenizer via HuggingFace
		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		self.model = AutoModelForCausalLM.from_pretrained(
			MODEL_NAME,
			# Laisse transformer choisir la précusion numérique des poids
			torch_dtype="auto",
			# Place automatiquement le modèle soit sur le GPU si présent, ou sur le CPU
			device_map="auto"
		)

	# CETTE FONCTION DOIT PRENDRE LES CHUNKS DONNEES DE SEARCH() ET LES ASSEMBLER ENSEMBLE
	def build_context(self, chunks: list[ChunkData]) -> str:
		result = []
		for chunk in chunks:
			# Le chemin du fichier permet au modele de citer sa source.
			result.append(f"[{chunk.file_path}]\n{chunk.text}")
		return "\n\n".join(result)

	def find_chunks(
		self,
		sources: list[MinimalSource],
		chunks: list[ChunkData],
	) -> list[ChunkData]:
		by_span = {
			(c.file_path, c.first_character_index, c.last_character_index): c
			for c in chunks
		}
		found = []
		for source in sources:
			key = (
				source.file_path,
				source.first_character_index,
				source.last_character_index,
			)
			if key in by_span:
				found.append(by_span[key])
		return found

	def answer_questions(
		self,
		search_results: StudentSearchResults,
		chunks: list[ChunkData],
	) -> StudentSearchResultsAndAnswer:
		# Un modele pydantic se construit d'un bloc, avec tous ses champs :
		# on accumule d'abord dans une liste, on construit le modele a la fin.
		answers = []
		for result in tqdm(search_results.search_results, desc="answering"):
			retrieved = self.find_chunks(result.retrieved_sources, chunks)
			context = self.build_context(retrieved)
			answer = MinimalAnswer(
				question_id=result.question_id,
				question=result.question,
				retrieved_sources=result.retrieved_sources,
				answer=self.generate_answer(result.question, context),
			)
			answers.append(answer)
		return StudentSearchResultsAndAnswer(
			search_results=answers,
			k=search_results.k,
		)

	def generate_answer(self, question: str, context: str) -> str:
		if context:
			user_content = f"Sources:\n{context}\n\nQuestion: {question}"
		else:
			user_content = (
				f"Question: {question}\n\n"
				"No source was retrieved for this question."
			)
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": user_content},
		]
		text = self.tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False
		)
		model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

		# ici il écrit le text de réponse
		with torch.inference_mode():
			generated_ids = self.model.generate(
				**model_inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				do_sample=False,
				pad_token_id=self.tokenizer.eos_token_id,
			)
		output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1]:]
		content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
		return content


# Test : ne s'execute que via `python -m src.generator`, jamais a l'import.
if __name__ == "__main__":
	import time
	from pathlib import Path
	from .indexer import load_index
	from .retriever import load_questions, search_dataset

	bm25, chunks = load_index(Path("data/processed/index.pkl"))
	dataset = load_questions(
		Path("data/datasets/UnansweredQuestions/dataset_code_public.json")
	)
	# 3 questions suffisent pour verifier que tout s'enchaine.
	dataset.rag_questions = dataset.rag_questions[:3]
	results = search_dataset(dataset, bm25, chunks, k=5)

	test = Questions()
	test.load_model()
	start = time.time()
	answered = test.answer_questions(results, chunks)
	print(f"\n{len(answered.search_results)} reponses en {time.time() - start:.1f}s\n")
	for a in answered.search_results:
		print("Q:", a.question)
		print("A:", a.answer)
		print()
