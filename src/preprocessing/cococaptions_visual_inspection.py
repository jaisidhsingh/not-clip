import torch
import ast

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

results = torch.load('/workspace/datasets/coco/coco_helper_val_image_captions_final.pt')

for i in range(50):
    print('decomposition:', extract_answer(results['decomposition'][i]))
    print('P      :', results['P']    [i])
    print('N1     :', results['N1']   [i])
    print('C^R_1  :', results['C^R_1'][i])
    print('N2     :', results['N2']   [i])
    print('-'*25)
    print()