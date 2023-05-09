import pickle
from collections import defaultdict
import random
import json


from util import mention2task

random.seed(42)

mention_map = pickle.load(open('./data/mention_map.pkl', 'rb'))

evt_mention_map_dev = {m_id: men for m_id, men in mention_map.items() if men['split'] == 'dev' and men['men_type'] == 'evt'}

cluster2men = defaultdict(list)
singleton2men = {}

topic2cluster = defaultdict(list)
topic2singles = defaultdict(list)

for men in evt_mention_map_dev.values():
    gold_cluster = str(men['gold_cluster'])
    if gold_cluster.startswith('ACT') or gold_cluster.startswith('NEG'):
        cluster2men[men['gold_cluster']].append(men)
    else:
        singleton2men[men['gold_cluster']] = men
pass


cluster2men = {clus: mens for clus, mens in cluster2men.items() if len(mens) >= 5}

for clus, mens in cluster2men.items():
    men = mens[0]
    topic = men['topic']
    topic2cluster[topic].append(clus)

for clus, men in singleton2men.items():
    topic = men['topic']
    topic2singles[topic].append(men)


all_task_mentions = []

for topic in topic2cluster:
    topic_clusters = topic2cluster[topic]
    topic_singletons = topic2singles[topic]

    topic_clusters = sorted(topic_clusters, key=lambda x: len(cluster2men[x]), reverse=True)

    best_clusters = topic_clusters[:2]

    topic_singletons = random.sample(topic_singletons, k=3)

    topic_cluster_mentions = []

    for clus in best_clusters:
        topic_cluster_mentions.extend(random.sample(cluster2men[clus], k=6))

    all_task_mentions.extend(topic_singletons)
    all_task_mentions.extend(topic_cluster_mentions)

imp_keys = ['doc_id', 'sentence_id', 'mention_id', 'gold_cluster', 'lemma', 'topic', 'pos', 'bert_doc']
tasks = [mention2task(men, imp_keys) for men in all_task_mentions]

for task in tasks:
    task['bert_doc'] = task['bert_doc'].replace('\n', '<p>')

json.dump(tasks, open('./data/dev_iter.json', 'w'), indent=1)

with open('test.html', 'w') as tfw:
    tfw.write('<p>')
    tfw.write(tasks[0]['bert_doc'])
    tfw.write('</p>')
pass

