#pip install --upgrade openai
import os
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)
from openai.api_resources import (
    Completion,
    Engine,
    ErrorObject,
    File,
    Answer,
    Classification,
    Snapshot,
    FineTune
)

# Load your API key from an environment variable or secret management service
def chat_gpt(prompt):
    # 你的问题
    prompt = prompt

    # 调用 ChatGPT 接口
    model_engine = "text-davinci-003"
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    print(response)

chat_gpt("Python怎么从入门到精通，具体的学习方法是什么？")

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())
#
# openai.api_key  = os.getenv('OPENAI_API_KEY')
#
# def get_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.Completion.create(
#         model=model,
#         messages=messages,
#         temperature=0, # this is the degree of randomness of the model's output
#     )
#     return response.choices[0].message["content"]
#
# ## Write clear and specific instructions
# text = f"""
# You should express what you want a model to do by \
# providing instructions that are as clear and \
# specific as you can possibly make them. \
# This will guide the model towards the desired output, \
# and reduce the chances of receiving irrelevant \
# or incorrect responses. Don't confuse writing a \
# clear prompt with writing a short prompt. \
# In many cases, longer prompts provide more clarity \
# and context for the model, which can lead to \
# more detailed and relevant outputs.
# """
# prompt = f"""
# Summarize the text delimited by triple backticks \
# into a single sentence.
# ```{text}```
# """
# response = get_completion(prompt)
# print(response)