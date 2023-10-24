import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import torch
from tqdm import tqdm
import os
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gptq_model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
gptq_model = AutoModelForCausalLM.from_pretrained(gptq_model_id, device_map="auto")
gptq_tokenizer = AutoTokenizer.from_pretrained(gptq_model_id)
gptq_tokenizer.pad_token = gptq_tokenizer.eos_token


def generate(gptq_model, gptq_tokenizer, prompt):
    inputs = gptq_tokenizer(prompt, return_tensors='pt').to(device)
    out = gptq_model.generate(
        **inputs,
        temperature=0.5,
        do_sample=True, 
        top_p=0.95, 
        top_k=10, # 40 was fab. maybe 10 is enough.
        max_new_tokens=256,
    )
    answer = gptq_tokenizer.decode(out[0])
    answer = answer.lower().split('answer:')[-1].replace('</s>', '')
    return answer


def get_subject(gptq_model, gptq_tokenizer, sentence):
    pre_text = 'In this game, you have to decompose a given sentence into subject, objects and their associated prepositions, and re-write only the subject thus obtained.'
    example = 'For example: "a flag without a dome" -> "a flag".'
    concise = 'Be as succinct as possible i.e. do not give any explanations or details apart from the answer asked. '
    def template(sentence):
        return f'{concise} {pre_text} {example} Similarly, give me the subject of the given sentence: "{sentence}". Answer:'
    subject = generate(gptq_model, gptq_tokenizer, template(sentence))
    return subject


def get_objects(gptq_model, gptq_tokenizer, sentence, subject):
    pre_text = 'In this game, you are given a sentence and its subject. You have to get all the other objects in the sentence and re-write them in a list.'
    example = '''For example: 
GIVEN: {sentence:"a person on skis makes her way through the snow", subject:"a person on skis"} -> ANSWER:
- "the snow"
'''
    concise = 'Be as succinct as possible i.e. do not give any explanations or details apart from the answer asked. '
    thought = 'Keep in mind that the predicate is the relation between the subject and the objects.'
    def template(sentence, subject):
        p1 = f'{concise} {pre_text} {example} Similarly, give me the objects of the given sentence:\n'
        p2 = f'GIVEN: '+'{sentence:'+f'"{sentence}", subject:"{subject}"'+'} -> ANSWER:\n'
        return p1 + p2 + thought
    print(template(sentence, subject))
    objects = generate(gptq_model, gptq_tokenizer, template(sentence, subject))
    return objects

def get_objects_new(gptq_model, gptq_tokenizer, sentence):
    llama_template = lambda prompt: f'''[INST] <<SYS>>
You are a helpful and honest assistant. Always answer as helpfully as possible. Please don't share false information.
<</SYS>>
{prompt}[/INST]'''
    template_text = '''In this game, we are required to decompose a sentence into its subject-predicate-object structure. An example of this is shown as follows:
EXAMPLE1:
sentence:"a person makes her way through the snow"
subject:"a person"
objects_predicates:[{"object": "the snow", "predicate": "makes her way through"}] 
(explanation: the relation between the subject "a person" and the object "snow")


Keep in mind that the predicate is the relation between the subjects and the objects. thus, every object must be associated with one predicate. This means that the length of the object lists and the predicate lists have to be the same, even if duplication of elements is required.
Remember that every object must have a predicate associated with it (both lists must be of the same size.)

Understand the steps from this example and use the following answer format (without the explanations given above):
QUESTION:
>sentence:"a person on skis makes her way through the snow"
ANSWER:
>subject:"a person"
>objects_predicates:[{"object":"the snow", "predicate":"makes her way through"}, {"object":"skis", "predicate":"on"}]

Now, answer the following question while being as succinct as possible (no explanations or details apart from the ones I asked for)-
QUESTION:
>sentence:'''
    def template(sentence):
        return llama_template(template_text + sentence)
    
    return generate(gptq_model, gptq_tokenizer, template(sentence))


##############################

sentence = 'A woman stands in the dining area at the table.'
# subject = get_subject(gptq_model, gptq_tokenizer, sentence)
# subject = subject.strip(' ."')
# # print(subject)
# objects = get_objects(gptq_model, gptq_tokenizer, sentence, subject)
# print(objects)

stuff = get_objects_new(gptq_model, gptq_tokenizer, sentence).strip()
print(stuff)

##########

input_ptfile = '/workspace/datasets/coco/coco_helper_val_image_captions.pt'
output_ptfile = '/workspace/datasets/coco/coco_helper_val_image_captions_decomposed.pt'

results = torch.load(input_ptfile)
new_results = {'image_paths': [], 'captions': [], 'chosen_caption': [], 'intermediate_output': []}

bar = tqdm(total=len(results['captions']))
for image_path, captions in zip(results['image_paths'], results['captions']):
    new_results['image_paths'].append(image_path)
    new_results['captions'].append(captions)

    chosen_caption = max(captions, key=lambda x: len(x))
    new_results['chosen_caption'].append(chosen_caption)

    intermediate_output = get_objects_new(gptq_model, gptq_tokenizer, chosen_caption).strip()
    new_results['intermediate_output'].append(intermediate_output)

    bar.update(1)
    print(f'sentence: {chosen_caption}')
    print(intermediate_output)
    print('-'*25)
    torch.save(new_results, output_ptfile)
bar.close()

torch.save(new_results, output_ptfile)
    