import os

# transformers 5 charge les poids du modele avec 4 threads. Sur macOS, cela
# provoque un deadlock intermittent dans safetensors (PySafeSlice.__getitem__
# -> OnceLock::initialize, observe a la pile d'appel) : le processus reste
# bloque a 0 % CPU indefiniment. Le chargement sequentiel est fiable et, pour
# un modele de 0.6B, pas plus lent. La variable doit etre positionnee AVANT
# le premier import de transformers : le __init__ du paquet est le seul
# endroit qui s'execute a coup sur avant tous les sous-modules.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
