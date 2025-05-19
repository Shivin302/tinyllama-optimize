import requests
import json
import time

def test_generate():
    url = "http://localhost:8001/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "Once upon a time in a land far, far away",
        "max_tokens": 100,
        "top_p": 0.9,
        "top_k": 50
    }
    
    print("Sending request to Transformers API...")
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print("\nGenerated Text:")
        print(result["generated_text"])
        print("\nMetrics:")
        metrics = result["metrics"]
        print(f"Prompt tokens: {metrics['prompt_tokens']}")
        print(f"Generated tokens: {metrics['generated_tokens']}")
        print(f"Total time: {metrics['total_time']:.2f}s")
        print(f"Tokens per second: {metrics['tokens_per_second']:.2f}")
    else:
        print(f"Error {response.status_code}:")
        print(response.text)
    
    print(f"\nTotal request time: {end_time - start_time:.2f} seconds")

def test_health():
    url = "http://localhost:8001/"
    response = requests.get(url)
    print("\nHealth Check:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing Transformers API for TinyLlama")
    print("=" * 40)
    
    # Test health check first
    try:
        test_health()
        
        # Test text generation
        print("\nTesting text generation...")
        test_generate()
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server. Make sure it's running.")
        print("Start the server with: python app.py")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
