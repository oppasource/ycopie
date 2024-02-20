import pdb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
# model_name = "tiiuae/falcon-180B-chat"
model_path = "/data/base_models/"

# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path, 
#                                                     device_map="auto", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path, 
                        device_map="auto", use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)


def get_llm_reponse(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, top_k=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def post_process_response(response, prompt):
    prcessed_response = response[len(prompt):].strip()
    return prcessed_response


def get_response_to_prompt(prompt):
    response = get_llm_reponse(model, tokenizer, prompt, max_new_tokens=500)
    response = post_process_response(response, prompt)
    return response


if __name__ == "__main__":
    prompt = "Google is "
    print(get_response_to_prompt(prompt))
    pdb.set_trace()




