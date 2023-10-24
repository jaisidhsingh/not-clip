import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import torch
from tqdm import tqdm
import os
import sys
import ast

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gptq_model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
gptq_model = AutoModelForCausalLM.from_pretrained(gptq_model_id, device_map="auto")
gptq_tokenizer = AutoTokenizer.from_pretrained(gptq_model_id)
gptq_tokenizer.pad_token = gptq_tokenizer.eos_token
allowed_predicates = ['not', 'without']


def generate(gptq_model, gptq_tokenizer, prompt):
    inputs = gptq_tokenizer(prompt, return_tensors='pt').to(device)
    out = gptq_model.generate(
        **inputs,
        temperature=0.5,
        do_sample=True, 
        top_p=0.95, 
        top_k=10, # 40 was fab. maybe 10 is enough.
        max_new_tokens=128,
    )
    answer = gptq_tokenizer.decode(out[0])
    answer = answer.lower().split('answer:')[-1].replace('</s>', '')
    return answer

def get_negation(gptq_model, gptq_tokenizer, sentence, subject, objects, predicates, chosen_object):
    template_text = f'''In this game, We are required to change the meaning of a sentence by modifying the relation between the subject and one of the objects by chaning the predicate associating the two. 
The new predicate must show negation or non-association or non-belonging of the object to the subject, and must use one of the words: {allowed_predicates}
Contractions like "doesn't" "hasn't" "shouldn't" should be replaced with their original words.

An example of this is shown as follows:
EXAMPLE1:
```
QUESTION
>sentence:"a person on skis makes her way through the snow"
>subject:"a person"
>objects:["the snow", "skis"]
>predicates:["makes her way through", "on"]
>chosen object:"snow"

ANSWER
>negative prompt:"a person on skis is not on the snow"
```

Explanation of the example: the predicate "makes her way through" between the subject "a person" and the object "the snow" is replaced. The other objects ["skis"] and their corresponding prepositions remain unchanged.
Keep in mind that the predicate is the relation between the subjects and the objects.

Similarly, here is another example, being as succinct as possible (no explanations or details)-

```
QUESTION:
>sentence:"{sentence}"
>subject:"{subject}"
>objects:{objects}
>predicates:{predicates}
>chosen object:"{chosen_object}"

ANSWER:
>negative prompt:'''
    
    return generate(gptq_model, gptq_tokenizer, template_text)



input_ptfile = '/workspace/datasets/coco/coco_helper_val_image_captions_decomposed.pt'
output_ptfile = '/workspace/datasets/coco/coco_helper_val_image_captions_negations.pt'

results = torch.load(input_ptfile)
new_results = {'image_paths': [], 'image_index': [], 'captions': [], 'chosen_caption': [], 'subject': [], 'objects': [], 'predicates': [], 'chosen_object': [], 'negation': []}


bar = tqdm(total = len(results['image_paths']))
for image_path, captions, chosen_caption, intermediate_output, image_index in zip(results['image_paths'], results['captions'], results['chosen_caption'], results['intermediate_output'], range(len(results['image_paths']))):
    subject, objects, predicates = intermediate_output.split('\n')
    subject = ast.literal_eval(subject[len('>subject:'):])
    objects = ast.literal_eval(objects[len('>objects:'):])
    predicates = ast.literal_eval(predicates[len('>predicates:'):])
    for chosen_object, chosen_predicate in zip(objects, predicates):
        try:
            negation = get_negation(gptq_model, gptq_tokenizer, chosen_caption, subject, objects, predicates, chosen_object)
            new_results['image_paths'].append(image_path)
            new_results['image_index'].append(image_index)
            new_results['captions'].append(captions)
            new_results['chosen_caption'].append(chosen_caption)
            new_results['subject'].append(subject)
            new_results['objects'].append(objects)
            new_results['predicates'].append(predicates)
            new_results['chosen_object'].append(chosen_object)
            negation = negation.split('```')[0].strip().strip('>:qwertyuiopasdfghjklzxcvbnm ')
            new_results['negation'].append(negation)
        except Exception as e:
            print(f'''Error: Image Index:{image_index}''')
    bar.update(1)
bar.close()

torch.save(new_results, output_ptfile)

for chosen_object, negation in zip(helper['chosen_object'], helper['negation']):
    if chosen_object in negation:
            count += 1
    total += 1
    if chosen_object not in negation:
            print(f'chosen_object: {chosen_object}')
            print(f'negation     : {negation}')
            print()
    