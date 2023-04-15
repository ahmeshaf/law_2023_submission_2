from prodigy.components.loaders import JSONL
import numpy as np

adj_tasks = JSONL('./data/ecb_gpt_adj.jsonl')

accs = []
covs = []

for t in adj_tasks:
    usr_in = t['user_input']
    accs.append([int(i) for i in usr_in.split('|')[0].split(',')])
    covs.append([int(i) for i in usr_in.split('|')[-1].split(',')])


accs = np.array(accs)
covs = np.array(covs)

print('accuracy&&', '&'.join(['%.1f'%i for i in accs.mean(axis=0)]), '\\\\')
print('coverage&&',  '&'.join(['%.1f'%i for i in covs.mean(axis=0)]))