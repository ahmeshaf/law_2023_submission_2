import json
import pickle

from prodigy.components.loaders import JSONL, JSON
import re
from scipy.sparse import lil_matrix

from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
from util import generate_key_file
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components


def get_year(string_):
    years = re.findall(r'(\d{4})', string_)
    return years


def has_roleset(str_):
    rs = re.findall(r'.*\.\d{1,3}', str_)
    return len(rs) > 0


def resolve_rs(task, syn_map):
    rs = task['roleset_id']
    topic = task['topic']
    if (topic, rs) in syn_map:
        return topic, syn_map[topic, rs]
    else:
        return topic, rs


def clean_arg(arg_value):
    return arg_value.replace('NA', '').strip().replace('%27', "'").replace('_', ' ')


def add_eid(task, syn_map, sent_rs_id2task, human=False):
    rs_resolved = resolve_rs(task, syn_map)
    if human:
        arg0s = clean_arg(task['arg0']).split('/')
        arg1s = clean_arg(task['arg1']).split('/')
        argLs = clean_arg(task['argL']).split('/')
        argTs = clean_arg(task['argT']).split('|')
    else:
        arg0s = clean_arg(task['arg0']).split(' ')
        arg1s = clean_arg(task['arg1']).split(' ')
        argLs = clean_arg(task['argL']).split(' ')
        argTs = clean_arg(task['argT']).split('|')

    doc_id = task['doc_id']
    sentence_id = task['sentence_id']

    if 'EIDs' not in task:
        task['EIDs'] = set()
        # only standard eid
        for arg0 in arg0s:
            if (doc_id, sentence_id, arg0) in sent_rs_id2task:
                add_eid(sent_rs_id2task[doc_id, sentence_id, arg0][0], syn_map, sent_rs_id2task,human)
            for arg1 in arg1s:
                if (doc_id, sentence_id, arg1) in sent_rs_id2task:
                    add_eid(sent_rs_id2task[doc_id, sentence_id, arg1][0], syn_map, sent_rs_id2task, human)
                eid_curr = (arg0, rs_resolved, arg1)
                if (doc_id, sentence_id, arg1) not in sent_rs_id2task:
                    task['EIDs'].add(eid_curr)
                else:
                    hop_eids = set(sent_rs_id2task[(doc_id, sentence_id, arg1)][0]['EIDs'])
                    for eid in hop_eids:
                        for i in range(0, len(eid) - 2, 2):
                            task['EIDs'].add((arg0, rs_resolved) + eid[i:])
    if 'EIDsTime' not in task:
        task['EIDsTime'] = set()
        task['EIDsLoc'] = set()
        for arg0 in arg0s:
            for argt in argTs:
                if argt:
                    task['EIDsTime'].add((arg0, rs_resolved, argt))
                    if '2023' in argt:
                        argt = argt.replace('2023', '')
                    years = [argt] + get_year(argt)
                    for y in years:
                        if y:
                            task['EIDsTime'].add((arg0, rs_resolved, y))
            for argl in argLs:
                if argl:
                    task['EIDsLoc'].add((arg0, rs_resolved, argl))


def generate_eids(tasks, syn_map, human=False):
    sent_rs_id2task = {}
    for task in tasks:
        sent_rs_id = (task['doc_id'], task['sentence_id'], task['roleset_id'])
        if sent_rs_id not in sent_rs_id2task:
            sent_rs_id2task[sent_rs_id] = []
        sent_rs_id2task[sent_rs_id].append(task)

    for task in tasks:
        add_eid(task, syn_map, sent_rs_id2task, human)


def event_id_resolve(tasks, add_loc_time):
    mid2cluster = []
    eid2mid = {}

    for task in tasks:
        eids = list(list(task['EIDs']))
        if add_loc_time:
            eids = list(task['EIDsTime']) + list(task['EIDsLoc']) + list(task['EIDs'])
        # eids = list(task['EIDsTime']) + list(task['EIDsLoc'])
        for eid in eids:
            if eid not in eid2mid:
                eid2mid[eid] = []
            eid2mid[eid].append(task['mention_id'])

    m_ids = [t['mention_id'] for t in tasks]
    mid2i = {m: i for i, m in enumerate(m_ids)}
    n = len(m_ids)

    rs_mat = lil_matrix((n, n))

    for rs_cluster in eid2mid.values():
        for r1 in rs_cluster:
            for r2 in rs_cluster:
                rs_mat[mid2i[r1], mid2i[r2]] = 1

    adj_matrix = rs_mat.tocsr()

    # Find connected components
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    for rs, l in zip(m_ids, labels):
        mid2cluster.append((rs, l))

    return dict(mid2cluster)


def run_coreference_results(gold_clusters, predicted_clusters):
    gold_key_file = f'evt_gold.keyfile'
    generate_key_file(gold_clusters, 'evt', './', gold_key_file)
    system_key_file = './evt_annotated.keyfile'
    generate_key_file(predicted_clusters, 'evt', './', system_key_file)

    def read(key, response):
        return get_coref_infos('%s' % key, '%s' % response,
                               False, False, True)

    doc = read(gold_key_file, system_key_file)
    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cp, cr, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    # print('MUC', (mr, mp, mf))
    # print('B-CUB', (br, bp, bf))
    # print('CEAF', (cr, cp, cf))
    # print('CONLL', (mf+bf+cf)/3)

    recll = np.round((mr + br)/2, 1)
    precision = np.round((mp + bp)/2, 1)
    connl = np.round((mf + bf + cf) / 3, 1)

    print(' &&', recll, '&', precision, '&', connl, '&', lf)


def get_tasks(source_file, topics=None):

    if source_file.endswith('jsonl'):
        tasks = list(JSONL(source_file))
    else:
        tasks = list(JSON(source_file))

    if topics:
        return [t for t in tasks if t['topic'] in topics]
    else:
        return tasks


def resolve_dict(key2cluster_arr):
    key2cluster = {}
    all_keys = sorted(list(key2cluster_arr.keys()))
    key2i = {k: i for i, k in enumerate(all_keys)}
    n = len(all_keys)
    cluster2keys = {}
    for key, cluster in key2cluster_arr.items():
        for c in cluster:
            if c not in cluster2keys:
                cluster2keys[c] = []
            cluster2keys[c].append(key)

    key_mat = lil_matrix((n, n))

    for cluster in cluster2keys.values():
        for k1 in cluster:
            for k2 in cluster:
                key_mat[key2i[k1], key2i[k2]] = 1

    adj_matrix = key_mat.tocsr()

    # Find connected components
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    for k, l in zip(all_keys, labels):
        key2cluster[k] = l
    return key2cluster


def get_syn_map_hum(tsv_file):
    with open(tsv_file) as rsf_g:
        rs2cluster_arr = {}
        rows = [line.split('\t') for line in rsf_g][1:]
        for row in rows:
            cl = [c.strip() for c in row[0].split('|')]
            topic = row[1]
            cl = [(topic, c) for c in cl]
            rs_ = row[2]
            if rs_ not in rs2cluster_arr:
                rs2cluster_arr[topic, rs_] = set()
            rs2cluster_arr[topic, rs_].update(cl)

    syn_hum = resolve_dict(rs2cluster_arr)
    return syn_hum


def get_syn_map_vn():
    pb_dict = pickle.load(open('./data/roleset.dict', 'rb'))
    rs2cluster_arr = {}
    for rs, rs_dict in pb_dict.items():
        if rs not in rs2cluster_arr:
            rs2cluster_arr[rs] = []
            for lexlink in rs_dict['lexlinks']:
                if lexlink['@resource'] == 'VerbNet':
                    rs2cluster_arr[rs].append(lexlink['@class'][0].split('.')[0])
    syn_vn = resolve_dict(rs2cluster_arr)
    return syn_vn

# get_syn_map_vn()


def generate_results(my_tasks, add_loc_time=True, only_eventive=False, only_non_eventive=False):
    if only_eventive:
        curr_tasks = [task for task in my_tasks if has_roleset(task['arg1'])]
    elif only_non_eventive:
        curr_tasks = [task for task in my_tasks if not has_roleset(task['arg1'])]
    else:
        curr_tasks = my_tasks
    mid2task = {task['mention_id']: task for task in curr_tasks}
    gold_clusters_ = [(t['mention_id'], t['gold_cluster']) for t in curr_tasks]
    predicted_ = event_id_resolve(curr_tasks, add_loc_time=add_loc_time)
    predicted_clusters_ = [(t['mention_id'], str(predicted_[t['mention_id']])) for t in curr_tasks]
    run_coreference_results(gold_clusters_, predicted_clusters_)


def generate_results_rs_only(my_tasks, syn_map=None, oracle=False):
    gold_clusters_ = [(t['mention_id'], t['gold_cluster']) for t in my_tasks]
    predicted_clusters_ = [(t['mention_id'], (t['topic'], t['roleset_id'].lower())) for t in my_tasks]
    if syn_map:
        predicted_clusters_ = [(t['mention_id'], resolve_rs(t, syn_map)) for t in my_tasks]

    m2task = {t['mention_id']: t for t in my_tasks}

    if oracle:
        predicted_clusters_ = [(m_id, clus_id + (m2task[m_id]['gold_cluster'],)) for m_id, clus_id in predicted_clusters_]
    # predicted_clusters_ = [(t['mention_id'], str(predicted_[t['mention_id']])) for t in my_tasks]
    run_coreference_results(gold_clusters_, predicted_clusters_)


def generate_results_lemma(my_tasks):
    gold_clusters_ = [(t['mention_id'], t['gold_cluster']) for t in my_tasks]
    predicted_clusters_ = [(t['mention_id'], (t['topic'], t['lemma'].lower())) for t in my_tasks]
    # if syn_map:
    #     predicted_clusters_ = [(t['mention_id'], resolve_rs(t, syn_map)) for t in my_tasks]
    # predicted_clusters_ = [(t['mention_id'], str(predicted_[t['mention_id']])) for t in my_tasks]
    run_coreference_results(gold_clusters_, predicted_clusters_)


def clean_gpt_response():
    import pickle

    gpt_tasks = pickle.load(open('./data/gpt4_ecb_test.pkl', 'rb'))
    gpt_tasks_clean = []
    for m_id, task in gpt_tasks.items():
        if 'api_key' in task:
            task.pop('api_key')
        if 'api_key' in task['usage']:
            task['usage'].pop('api_key')

        if 'api_key' in task['choices'][0]:
            task['choices'][0].pop('api_key')

        if 'api_key' in task['choices'][0]['message']:
            task['choices'][0]['message'].pop('api_key')

        new_task = dict(task)
        new_task['usage'] = dict(new_task['usage'])
        new_task['choices'][0] = dict(new_task['choices'][0])
        new_task['choices'][0]['message'] = dict(new_task['choices'][0]['message'])
        new_task['mention_id'] = m_id
        # new_task = dict(task)
        gpt_tasks_clean.append(new_task)

    json.dump(gpt_tasks_clean, open('./data/gpt4_output.json', 'w'), indent=1)


def create_gpt_prodigy_tasks():
    from random import sample
    from random import seed
    from prodigy.util import split_string, set_hashes
    seed(42)
    from collections import defaultdict
    gpt_tasks = json.load(open('./data/gpt4_output.json'))
    men2gpt_task = {t['mention_id']:t for t in gpt_tasks}
    all_tasks = json.load(open('./data/annotations/adjudication_clean.json'))
    topic2m_id = defaultdict(list)
    men2task = {t['mention_id']: t for t in all_tasks}
    for task in all_tasks:
        gold_cluster = str(task['gold_cluster'])
        mention_id = task['mention_id']
        if gold_cluster.startswith('ACT'):
            topic2m_id[task['topic']].append(mention_id)

    random_t_m_ids = []
    for val in topic2m_id.values():
        random_t_m_ids.extend(sample(val, k=10))
    print(random_t_m_ids)
    # print(len(random_t_m_ids))

    def parse_response(gpt_response):
        lines = gpt_response.split('\n')
        roleset = ''
        arg0 = ''
        arg1 = ''
        argL = ''
        argT = ''
        for line in lines:
            if line.startswith('roleset'):
                roleset = line.split('roleset:')[-1].strip()
            elif line.startswith('ARG-0'):
                arg0 = line.split('ARG-0:')[-1].strip()
            elif line.startswith('ARG-1'):
                arg1 = line.split('ARG-1:')[-1].strip()
            elif line.startswith('ARG-Location'):
                argL = line.split('ARG-Location:')[-1].strip()
            elif line.startswith('ARG-Time'):
                argT = line.split('ARG-Time:')[-1].strip()

        return roleset, arg0, arg1, argL, argT

    gpt_tasks_prodigy = []
    for m_id in random_t_m_ids:
        gpt_t = men2gpt_task[m_id]
        my_task = men2task[m_id]
        # my_task.pop('answer')
        r, a0, a1, aL, aT = parse_response(gpt_t['choices'][0]['message']['content'])
        ARG_NS = ['roleset_id', 'arg0', 'arg1', 'argL', 'argT']
        for a_name, a_val in zip(ARG_NS, [r, a0, a1, aL, aT]):
            my_task[a_name] = a_val
        my_task = set_hashes(my_task, ['text', 'spans'], overwrite=True)
        gpt_tasks_prodigy.append(my_task)

    json.dump(gpt_tasks_prodigy, open('./data/annotations/gpt_tasks_adjudication.json', 'w'))
    print(len(gpt_tasks_prodigy))


def final_results(only_eventive=False, only_non_eventive=False):
    m_topics = ['41', '42', '43', '44', '45']
    m_topics = None
    mention_map = pickle.load(open('./data/mention_map.pkl', 'rb'))
    # my_tasks = get_tasks('./data/annotations/ecb_evi_george_r2.jsonl', 'evi')
    my_tasks_hum = get_tasks('./data/annotations/adjudication_clean.json', m_topics)
    my_tasks_gpt = get_tasks('./data/annotations/gpt4-coref.json', m_topics)
    all_men_ids = list([t['mention_id'] for t in my_tasks_hum])
    eventive_ids = list([t['mention_id'] for t in my_tasks_hum if has_roleset(t['arg1'])])
    non_eventive_ids = list([t['mention_id'] for t in my_tasks_hum if not has_roleset(t['arg1'])])

    # print('all events', len(all_men_ids))
    # print('eventive events', len(eventive_ids))
    # print('non-eventive events', len(non_eventive_ids))
    for t in my_tasks_hum:
        t['lemma'] = mention_map[t['mention_id']]['lemma']
    for t in my_tasks_gpt:
        # t['roleset_id'] = mention_map[t['mention_id']]['lemma'].lower()
        t['roleset_id'] = t['roleset_id'].lower().split('.')[0]
    # generate_results_lemma(my_tasks_hum)
    # if only_eventive:
    my_tasks_hum_e = [t for t in my_tasks_hum if t['mention_id'] in eventive_ids]
    my_tasks_gpt_e = [t for t in my_tasks_gpt if t['mention_id'] in eventive_ids]
    # elif only_non_eventive:
    my_tasks_hum_ne = [t for t in my_tasks_hum if t['mention_id'] in non_eventive_ids]
    my_tasks_gpt_ne = [t for t in my_tasks_gpt if t['mention_id'] in non_eventive_ids]
    # my_tasks = list(JSONL('./data/annotations/evi.json'))
    # my_tasks = list(JSON('./data/annotations/evi.json'))

    pb_syn_hum = get_syn_map_hum('./data/annotations/roleset_syn_detection2.tsv')
    pb_syn_vn = get_syn_map_vn()
    for t in my_tasks_hum + my_tasks_gpt:
        topic = t['topic']
        roleset_id = t['roleset_id']
        if roleset_id in pb_syn_vn:
            pb_syn_vn[topic, roleset_id] = (topic, pb_syn_vn[roleset_id])
    # pb_syn_hum = pb_syn_vn
    pb_syn_hum = {}
    # pb_syn_vn={}
    generate_eids(my_tasks_hum, pb_syn_vn, human=True)
    generate_eids(my_tasks_gpt, pb_syn_vn, human=False)
    # print('rs only')
    syn_name = '+ \\PBSynHum'
    syn_name = ''

    def human_results():
        print('\nHuman')

        print(f'&\\RSHum {syn_name}')
        # generate_results_rs_only(my_tasks_hum_ne, pb_syn_vn)
        # generate_results_rs_only(my_tasks_hum_e, pb_syn_vn)
        generate_results_rs_only(my_tasks_hum, pb_syn_vn)
        print('\\\\')
        print(f'&\\RSHum {syn_name} + \\eid')
        # print('human')
        # generate_results(my_tasks_hum_ne, add_loc_time=False, only_eventive=False, only_non_eventive=True)
        # generate_results(my_tasks_hum_e, add_loc_time=False, only_eventive=True, only_non_eventive=False)
        generate_results(my_tasks_hum, add_loc_time=False, only_eventive=False, only_non_eventive=False)
        print('\\\\')
        print(f'&\\RSHum {syn_name} + \\eidLT')
        # generate_results(my_tasks_hum_ne, add_loc_time=True, only_eventive=False, only_non_eventive=True)
        # generate_results(my_tasks_hum_e, add_loc_time=True, only_eventive=True, only_non_eventive=False)
        generate_results(my_tasks_hum, add_loc_time=True, only_eventive=False, only_non_eventive=False)
        print('\\\\')
    human_results()
    print('\nGPT')
    print(f'&Lemma {syn_name}')
    # generate_results_rs_only(my_tasks_gpt_ne, pb_syn_vn)
    # generate_results_rs_only(my_tasks_gpt_e, pb_syn_vn)
    generate_results_rs_only(my_tasks_gpt, pb_syn_vn)
    print('\\\\')
    print(f'&Lemma {syn_name} + \\eid')
    # generate_results(my_tasks_gpt_ne, add_loc_time=False, only_eventive=False, only_non_eventive=True)
    # generate_results(my_tasks_gpt_e, add_loc_time=False, only_eventive=True, only_non_eventive=False)
    generate_results(my_tasks_gpt, add_loc_time=False, only_eventive=False, only_non_eventive=False)
    print('\\\\')
    print(f'&Lemma {syn_name} + \\eidLT')
    # generate_results(my_tasks_gpt_ne, add_loc_time=True, only_eventive=False, only_non_eventive=True)
    # generate_results(my_tasks_gpt_e, add_loc_time=True, only_eventive=True, only_non_eventive=False)
    generate_results(my_tasks_gpt, add_loc_time=True, only_eventive=False, only_non_eventive=False)
    print('\\\\')


print('all')
# final_results()

# print('\nonly-eventive')
# # final_results(only_eventive=True)
#
# print('\nonly-non-eventive')
# # final_results(only_non_eventive=True)
