from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"

print(f"Attempting to download model and tokenizer for: {model_name}")

try:
    # This line will download all necessary files to the correct cache location
    AutoTokenizer.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    print("\n--- Download complete! ---")
except Exception as e:
    print(f"\n--- An error occurred ---")
    print(f"{e}")
    print("\nPlease ensure you are logged in via 'huggingface-cli login' and have accepted the model's terms of use on its Hugging Face page.")

