import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, pipeline
import torch
from tqdm import tqdm
import os
import sys
import ast

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
allowed_predicates = ['not', 'without']
pipe = pipeline("text-generation", model="TheBloke/Mistral-7B-OpenOrca-GPTQ", device_map="auto")


def decompose_sentence(pipe, sentence):
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a helpful chatbot who likes to correctly decompose english sentences into their subject, objects and predicates. Note that objects must be nouns and predicates must be the actions linking the objects with the subject.",
        },
        {
            "role": "user",
            "content": """Fill in the empty ''s in this template by decomposing the sentence I am about to give you into subject, objects and predicates.
Here is the template for the output.
TEMPLATE:
>sentence:""
ANSWER:
>subject:""
>objects_predicates:[
    {"object":"", "predicate":""},
]
"""
        },
        {
            "role": "assistant",
            "content": "Understood. Please give me a sentence to decompose into this template. I will provide it in the same format as the example above."
        },
        {
            "role": "user",
            "content": f"Decompose this sentence: '{sentence}'"
        }
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.3, top_k=10, top_p=0.95)
    return (outputs[0]["generated_text"])

def extract_answer(generated_text):
    answer = generated_text.split('assistant')[-1]
    # subject
    start_index = answer.lower().index('subject:')
    while answer[start_index] != ':':
        start_index += 1
    start_index += 1
    final_index = start_index
    while answer[final_index] != '\n':
        final_index += 1
    subject = answer[start_index:final_index+1].strip()
    if subject[0] in ['"', "'"] and subject[-1] in ['"', "'"]:
        subject = ast.literal_eval(subject)
    #
    # objects, predicates
    start_index = answer.lower().index('predicates:')
    final_index = start_index
    while answer[start_index] != '[':
        start_index += 1
    while answer[final_index] != ']':
        final_index += 1
    objects_predicates = ast.literal_eval(answer[start_index:final_index+1])
    #
    return subject, objects_predicates

def negate_object(pipe, sentence, subject, objects_predicates, chosen_object):
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful chatbot that constructs sentences from subjects, objects and predicates. You will be given this information and you will be told to negate one of the predicates with a new predicate which has to be one of or similar to: {allowed_predicates}",
        },
        {
            "role": "user",
            "content": """negate the chosen_object_for_negation by changing its predicate to the appropriate one/variant of""" + str(allowed_predicates) + """. Then, recombine the new subject object predicate information into a new sentence. Here is an example:
EXAMPLE:
INPUT:
>subject:'a flag'
>objects_predicates:[{'object': 'dome', 'predicate':'on'}]
>chosen_object_for_negation:'dome'
OUTPUT:
>new_objects_predicates:[{'object': 'dome', 'predicate': 'without'}]
>new_sentence:'a flag without a dome'
"""
        },
        {
            "role": "assistant",
            "content": "Understood. Please give me the subject object predicate information to construct into a new sentence. I will provide it in the same format as the example above."
        },
        {
            "role": "user",
            "content": f"""
INPUT:
>subject:'{subject}'
>objects_predicates:{objects_predicates}
>chosen_object_for_negation:'{chosen_object}'
"""
        }
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.3, top_k=10, top_p=0.95)
    return (outputs[0]["generated_text"])

def extract_negation(negation):
    answer = negation.split('assistant')[-1]
    # negation
    start_index = answer.index('sentence:')
    while answer[start_index] != ':':
        start_index += 1
    start_index += 1
    final_index = start_index+1
    while final_index < len(answer) and answer[final_index] not in ["'", '"', '\n']:
        final_index += 1
    negation = answer[start_index:final_index+1].strip()
    if negation[0] in ['"', "'"] and negation[-1] in ['"', "'"]:
        negation = ast.literal_eval(negation)

    return negation

def neutralize_object(pipe, sentence, subject, objects_predicates):
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful chatbot that constructs sentences from subjects, objects and predicates. You will be given this information and you will be told to compose it into a sentence. If only the subject is given, then the output sentence should be just that.",
        },
        {
            "role": "user",
            "content": """recombine the subject object predicate information into a new sentence. Here is an example:
EXAMPLE:
INPUT:
>subject:'a flag'
>objects_predicates:[{'object': 'dome', 'predicate':'on'}, {'object':'temple', 'predicate':'of'}]
OUTPUT:
>new_sentence:'a flag on the dome of the temple'
"""
        },
        {
            "role": "assistant",
            "content": "Understood. Please give me the subject object predicate information to construct into a new sentence. I will provide it in the same format as the example above."
        },
        {
            "role": "user",
            "content": f"""
INPUT:
>subject:'{subject}'
>objects_predicates:{objects_predicates}
"""
        }
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.3, top_k=10, top_p=0.95)
    return (outputs[0]["generated_text"])

def extract_neutralization(neutral):
    answer = neutral.split('assistant')[-1]
    # negation
    start_index = answer.index('sentence:')
    while answer[start_index] != ':':
        start_index += 1
    start_index += 1
    final_index = start_index+1
    while final_index < len(answer) and answer[final_index] not in ["'", '"', '\n']:
        final_index += 1
    neutral = answer[start_index:final_index+1].strip()
    if neutral[0] in ['"', "'"] and neutral[-1] in ['"', "'"]:
        neutral = ast.literal_eval(neutral)

    return neutral

##########

input_ptfile = '/workspace/datasets/coco/coco_helper_val_image_captions.pt'
output_ptfile = '/workspace/datasets/coco/coco_helper_val_image_captions_final.pt'

results = torch.load(input_ptfile)
new_results = {
  'image_paths': [], 
  'captions': [], 
  'decomposition': [],
  'P': [],
  'N1': [],
  'N2': [],
  'C^R_1': [],
}

bar = tqdm(total=len(results['captions']))
max_error_count = 5
for image_index, (image_path, captions) in enumerate(list(zip(results['image_paths'], results['captions']))):
    error_count = max_error_count
    while error_count > 0:
      try:
        chosen_caption = max(captions, key=lambda x: len(x))

        #decomposition
        decomposition = decompose_sentence(pipe, chosen_caption)
        subject, objects_predicates = extract_answer(decomposition)
        #negation
        chosen_object_id = 0
        chosen_object = objects_predicates[chosen_object_id]['object']
        negation = negate_object(pipe, chosen_caption, subject, objects_predicates, chosen_object)
        #neutralization
        objects_predicates_new = []
        for i, op in enumerate(objects_predicates):
            if i != chosen_object_id:
                objects_predicates_new.append(op)
        neutralization = neutralize_object(pipe, chosen_caption, subject, objects_predicates_new)

        p = chosen_caption[::]
        n1 = extract_negation(negation)
        n2 = extract_neutralization(neutralization)
        cr1 = chosen_object[::]

        new_results['image_paths'].append(image_path)
        new_results['captions'].append(captions)
        new_results['decomposition'].append(decomposition)
        new_results['P'].append(p)
        new_results['N1'].append(n1)
        new_results['N2'].append(n2)
        new_results['C^R_1'].append(cr1)
        print(p)
        print(n1)
        print(objects_predicates_new)
        print(n2)

        bar.update(1)
        # torch.save(new_results, output_ptfile)
        break
      except Exception as e:
    #     print('error, retrying...')
    #     error_count -= 1
    #     if error_count <= 0:
    #       new_results['image_paths'].append(image_path)
    #       new_results['captions'].append(captions)
    #       new_results['decomposition'].append(decomposition)
    #       new_results['P'].append(None)
    #       new_results['N1'].append(None)
    #       new_results['N2'].append(None)
    #       new_results['C^R_1'].append(None)

        break

bar.close()

# torch.save(new_results, output_ptfile)
