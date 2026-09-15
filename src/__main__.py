import fire
from .cli import CLI
	

def main():
	try:
		fire.Fire(CLI)
	except KeyboardInterrupt:
		print("TA GRAND MERE A APPUYER SUR CTRL+C\n")
	except Exception as error:
		print(f"An error has occured: {error}\n")

if __name__ == "__main__":
	main()
		
