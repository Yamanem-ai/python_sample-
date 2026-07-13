import os
import urllib.request
import time
import ailia_llm

MODEL_FILE = "gemma-3-4b-it-Q4_K_M.gguf"

_llm = None


def initialize():

    global _llm

    if _llm is not None:
        return _llm

    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/ailia-models/gemma/gemma-3-4b-it-Q4_K_M.gguf",
            MODEL_FILE,
        )

    _llm = ailia_llm.AiliaLLM()
    _llm.open(MODEL_FILE)

    return _llm


def ask(text):
    try:
        model = ailia_llm.AiliaLLM()
        model.open(MODEL_FILE)
    
        messages = [
            {
                "role": "system",
                "content":
                "あなたは親切なAIアシスタントです。日本語で50文字以内で簡潔に回答してください。"
            },
            
            {
                "role": "user",
                "content": text
            }
        ]
    
        #print("messages =", messages)
        
        t0 = time.time()
        stream = model.generate(messages)

        response = ""

        for delta_text in stream:
            response += delta_text
        
        print(
            f"Gemma inference : "
            f"{time.time()-t0:.2f}s"
        )
        #print("context_full =", model.context_full())

        return response
    
    except Exception as e:
        return f"LLM Error: {e}"
        
def main():
    print("Gemma Ready")
    print("Type 'exit' to quit")
    
    while True:
        text = input("\nYou>")
        
        if text.lower() in ["exit", "quit"]:
            break
            
        answer = ask(text)
        
        print("\nGemma>")
        print(answer)

    
if __name__ == "__main__":
    main()
    

