"""
You can use this script to compute the accuracy given a dataset like
* generations_p=0.96.jsonl

and also a numpy array of machine/human probabilities that is the same size as the val and test sets.
"""
import json
import numpy as np
import os
import pandas as pd

# Load in the dataset
set_to_info = {'val': [], 'test': []}
with open('generator=mega~dataset=p0.94.jsonl', 'r') as f:
    for x in f:
        item = json.loads(x)
        if item['split'] == 'train':
            continue
        set_to_info[item['split']].append(item)

# Load in the probabilities

SAVED_PROBS_PATH='generator=mega~discriminator=grover~discsize=mega~dataset=p0.94~test-probs.npy'
assert os.path.exists(SAVED_PROBS_PATH)
probs = np.load(SAVED_PROBS_PATH)

############################ OK NOW HERE'S WHERE IT GETS INTERESTING
def score(probs, full_info):
    score_df = pd.DataFrame(data=probs, columns=['machine', 'human']) # THIS MUST AGREE
    score_df['labels'] = [x['label'] for x in full_info]
    score_df['orig_split'] = [x['orig_split'] for x in full_info]
    score_df['ind30k'] = [x['ind30k'] for x in full_info]
    score_df.index.name = 'raw_index'
    score_df.reset_index(inplace=True)

    # So really there are 3 groups here:
    # HUMAN WRITTEN ARTICLE (the "burner")
    # MACHINE WRITTEN ARTICLE PAIRED WITH HUMAN WRITTEN ARTICLE
    # For evaluation we want a 50:50 split between human and machine generations, meaning we need to take out the
    # burner part.
    groups = {k:v for k, v in score_df.groupby('orig_split')}
    unpaired_human = groups.pop('train_burner')

    machine_v_human = {k: v.set_index('ind30k', drop=True) for k, v in groups['gen'].groupby('labels')}
    machine_vs_human_joined = machine_v_human['machine'].join(machine_v_human['human'], rsuffix='_humanpair')
    machine_vs_human_joined['is_right'] = machine_vs_human_joined['machine'] > machine_vs_human_joined['machine_humanpair']

    combined_scores = pd.concat((
        unpaired_human[['machine', 'human', 'labels']],
        machine_vs_human_joined[['machine', 'human', 'labels']],
    ),0)
    combined_acc = np.mean(combined_scores[['machine', 'human']].idxmax(1) == combined_scores['labels'])

    stats = {
        'paired_acc': np.mean(machine_vs_human_joined['is_right']),
        'unpaired_acc': combined_acc,
    }
    return stats

# Compute the validation stats
val_stats = score(probs, set_to_info['test'])
print(val_stats)