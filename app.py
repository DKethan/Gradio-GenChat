import gradio as gr
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

model_name='microsoft/DialoGPT-large'
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name)

def predict(input,history=[]):
  new_user_input_ids=tokenizer.encode(input+tokenizer.eos_token,return_tensors='pt')
  bot_input_id=torch.cat([torch.LongTensor(history), new_user_input_ids],dim=-1)
  history=model.generate(bot_input_id, max_length=1000,pad_token_id=tokenizer.eos_token_id).tolist()
  response=tokenizer.decode(history[0]).split('<|endoftext|>')
  response=[(response[i],response[i+1]) for i in range (0,len(response)-1,2)]
  return response,history

run=gr.Interface(fn=predict,
                 inputs=["text","state"],
                 outputs=["chatbot","state"])
run.launch()

