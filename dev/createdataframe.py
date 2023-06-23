import os
import openai
import pandas as pd

openai.api_key = open("key.txt").read()

l_age = ['18', '20', '30', '40', '50', '60', '90']
l_gender = ['man', 'woman']
l_power = ['invisibility', 'read in the thoughts', 'turning lead into gold', 'immortality', 'telepathy', 'teleport', 'flight'] 

f_prompt = "Imagine a complete and detailed description of a {age}-year-old {gender} fictional character who has the superpower of {power}. Write out the entire description in a maximum of 100 words in great detail:"
f_sub_prompt = "{age}, {gender}, {power}"

df = pd.DataFrame()
for age in l_age:
 for gender in l_gender:
  for power in l_power:
   for i in range(3): ## 3 times each
    prompt = f_prompt.format(age=age, gender=gender, power=power)
    sub_prompt = f_sub_prompt.format(age=age, gender=gender, power=power)
    print(sub_prompt)

    response = openai.Completion.create(
     model="text-davinci-003",
     prompt=prompt,
     temperature=1,
     max_tokens=500,
     top_p=1,
     frequency_penalty=0,
     presence_penalty=0
    )
    
    finish_reason = response['choices'][0]['finish_reason']
    response_txt = response['choices'][0]['text']
    
    new_row = {
      'age':age, 
      'gender':gender, 
      'power':power, 
      'prompt':prompt, 
      'sub_prompt':sub_prompt, 
      'response_txt':response_txt, 
      'finish_reason':finish_reason}
    new_row = pd.DataFrame([new_row])
    df = pd.concat([df, new_row], axis=0, ignore_index=True)

print(df.head())
df.to_csv("out_openai_completion.csv")
