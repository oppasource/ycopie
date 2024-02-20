import uvicorn
import pdb
from fastapi import FastAPI, Request
from vllm import LLM, SamplingParams

model_name = "meta-llama/Llama-2-7b-chat-hf"

model_path = "/data/base_models"

# pdb.set_trace()

llm = LLM(model=model_name,
          download_dir=model_path,
          gpu_memory_utilization=0.7,
          tensor_parallel_size=4
         )

app = FastAPI()
    
@app.post("/ask")
async def generate_response_api(info: Request):
    
    req_info = await info.json()
    prompt = req_info.get('prompt')
    print("prompt",prompt)
    sampling_params = SamplingParams(temperature = 0, max_tokens = 500, n = 1)
    outputs = llm.generate(prompt, sampling_params)
    print("outputs",outputs)
    generated_response = outputs[0].outputs[0].text
    return {'answer':generated_response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
