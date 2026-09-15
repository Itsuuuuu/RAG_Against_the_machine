import fire

from .cli import CLI


def main() -> None:
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as error:
        # Filet de securite : un bug ne doit jamais faire remonter une
        # traceback a l'utilisateur, seulement un message.
        print(f"An error has occurred: {error}")


if __name__ == "__main__":
    main()
