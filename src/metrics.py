from .models import MinimalSource, RagDataset, StudentSearchResults

# Seuil de chevauchement retenu par la moulinette (sujet, VII.1.1).
IOU_THRESHOLD = 0.05
# Longueur maximale d'une source retournee (sujet, VI.1).
MAX_CONTEXT_LENGTH = 2000
DEFAULT_KS = (1, 3, 5, 10)


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    # Intersection over Union de deux intervalles [start, end).
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def is_found(reference: MinimalSource, retrieved: list[MinimalSource]) -> bool:
    # Une source de reference compte comme trouvee si l'un des resultats est
    # dans le MEME fichier et chevauche son intervalle (IoU >= seuil).
    for source in retrieved:
        if source.file_path != reference.file_path:
            continue
        iou = interval_iou(
            reference.first_character_index,
            reference.last_character_index,
            source.first_character_index,
            source.last_character_index,
        )
        if iou >= IOU_THRESHOLD:
            return True
    return False


def recall_at_k(
    results: StudentSearchResults, dataset: RagDataset, k: int
) -> float:
    # Recall@k d'une question = part de ses sources de reference retrouvees
    # dans le top-k. On moyenne sur toutes les questions du dataset.
    retrieved_by_id = {
        result.question_id: result.retrieved_sources[:k]
        for result in results.search_results
    }
    recalls = []
    for question in dataset.rag_questions:
        references = getattr(question, "sources", None)
        if not references:
            continue
        retrieved = retrieved_by_id.get(question.question_id, [])
        found = sum(1 for ref in references if is_found(ref, retrieved))
        recalls.append(found / len(references))
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def source_length(source: MinimalSource) -> int:
    return source.last_character_index - source.first_character_index


def count_oversized_sources(results: StudentSearchResults) -> int:
    # La moulinette rejette TOUT le fichier si une seule source depasse
    # max_context_length. Autant le savoir avant elle.
    return sum(
        1
        for result in results.search_results
        for source in result.retrieved_sources
        if source_length(source) > MAX_CONTEXT_LENGTH
    )


def evaluate_results(
    results: StudentSearchResults,
    dataset: RagDataset,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[int, float]:
    return {k: recall_at_k(results, dataset, k) for k in ks}
