import os
from dotenv import load_dotenv
from crew import MarketingCrew


def main():
    print("Initializing Smart Marketing Assistant...")

    # Load environment variables
    load_dotenv()

    # Check Ollama availability
    import urllib.request
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print(f"Ollama is running at {ollama_url}")
    except Exception as e:
        print(f"WARNING: Cannot connect to Ollama at {ollama_url}")
        print(f"Error: {e}")
        print("Please start Ollama first: ollama serve")
        return

    print(f"Using model: {model_name}")
    print("Starting Marketing Crew Execution (This may take several minutes)...")

    marketing_crew = MarketingCrew()
    result = marketing_crew.run()

    print("========================================")
    print("CREW EXECUTION COMPLETE!")
    print("========================================")
    print("The final marketing strategy has been saved to 'final-content-strategy.md'.")
    print("Preview of final output:")
    print(result)


if __name__ == "__main__":
    main()
