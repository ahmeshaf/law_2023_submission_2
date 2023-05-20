import copy
import glob
import random

from bs4 import BeautifulSoup
from prodigy.components.preprocess import add_tokens

# from events.parsing.parse_raw import docs2html
from util import HTML_INPUT, JAVASCRIPT_WSD, PB_HTML, DOC_HTML, DOC_HTML2, DO_THIS_JS_DISABLE, DO_THIS_JS
from tqdm.autonotebook import tqdm
from prodigy.components.loaders import JSONL, JSON
from prodigy.util import split_string, set_hashes
import numpy as np
from util import WhitespaceTokenizer
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
from scipy.spatial.distance import cosine
import requests
import pickle
from typing import Iterable, Optional, List
import os
import spacy
import prodigy
import unittest

STOPWORDS = {'the', 'a', 'an', 'of', 'is', 'was'}
ALL_ARGS = ['arg0', 'arg1', 'argL', 'argT']
ROLESET_ID = 'roleset_id'

stemmer = PorterStemmer()

MY_IP = requests.request('GET', 'https://checkip.amazonaws.com/').text.strip()


def get_propbank_dict(frames_folder='', force=False):
    if force or not os.path.exists('./data/roleset.dict'):
        roleset_dict = {}
        for frame in tqdm(glob.glob(frames_folder + '/*.xml'), desc='Reading FrameSet'):
            with open(frame) as ff:
                frame_bs = BeautifulSoup(ff.read(), parser='lxml', features="lxml")
                predicate = frame_bs.find('predicate')['lemma']
                rolesets = frame_bs.find_all('roleset')
                for roleset in rolesets:
                    roleset_dict[roleset['id']] = {
                        'id': roleset['id'],
                        'name': roleset['name'],
                        'frame': predicate,
                        'aliases': [
                            al.text for al in roleset.find_all('alias')
                        ],
                        'examples': [
                            [w for w in eg.find('text').text.split()
                             if w.lower() not in STOPWORDS]
                            for eg in roleset.find_all('example')
                        ]
                    }
        pickle.dump(roleset_dict, open('./data/roleset.dict', 'wb'))

    return pickle.load(open(frames_folder + './data/roleset.dict', 'rb'))


def get_max_repeated_element(list1):
    count_dict = dict(Counter(list1))
    if len(count_dict):
        max_repeated_element = max(count_dict, key=count_dict.get)
        return max_repeated_element
    return None


def make_wsd_tasks(stream, nlp, alias2roleset, propbank_dict,
                   sense2vec, field_suggestions, batch_size,
                   roleset2args, doc_port, pb_port):
    from random import shuffle

    texts = [(eg_["text"], eg_) for eg_ in stream]
    # shuffle(texts)

    pos_map = {'VERB': 'v', 'NOUN': 'n', 'JJ': 'j'}

    roleset_ids = list(propbank_dict.keys())

    for doc, eg_ in nlp.pipe(texts, batch_size=batch_size, as_tuples=True):
        task = copy.deepcopy(eg_)

        # task['field_suggestions'] = field_suggestions
        task_spans = task['spans']
        for span in task_spans:
            spacy_span = doc[span['token_start']: span['token_end'] + 1]
            span_root = spacy_span.root
            root_lemma = span_root.lemma_.lower()
            root_pos = span_root.pos_

            # lexeme = root_lemma.lower() + '.' + pos_map[root_pos]
            if span['label'] == 'EVT':
                task_per_span = copy.deepcopy(task)
                task_per_span['root_lemma'] = root_lemma
                task_per_span['spans'] = [copy.deepcopy(span)]
                task_per_span['options'] = []
                task_per_span['rolesetid'] = ''
                prop_holder = ''
                curr_roleset = ''
                if root_lemma in alias2roleset:
                    possib_rolesets = alias2roleset[root_lemma]
                    for roleset in possib_rolesets:
                        task_per_span['options'].append(
                            {'id': roleset, 'text': roleset + ": " + propbank_dict[roleset]['name']})
                        if roleset not in sense2vec:
                            sense2vec[roleset] = np.ones(doc.vector.shape)

                    if len(possib_rolesets) > 0:
                        cos_sims = [1 - cosine(sense2vec[rid], doc.vector) for rid in possib_rolesets]
                        # print(cos_sims)
                        best_roleset_id = possib_rolesets[np.argmax(cos_sims)]
                        task_per_span["accept"] = [propbank_dict[best_roleset_id]['id']]
                        curr_predicate = propbank_dict[best_roleset_id]['frame']
                        curr_roleset = propbank_dict[best_roleset_id]['id']
                        prop_holder = curr_predicate + '.html#' + curr_roleset
                        task_per_span['roleset_id'] = curr_roleset

                task_per_span['options'].append({'id': 'NONE', 'text': 'None of the above'})
                if 'accept' not in task_per_span:
                    task_per_span["accept"] = ["NONE"]

                # https://prodi.gy/docs/custom-recipes  # example-choice
                # task_per_span['prop'] = "http://0.0.0.0:8100/abandon.html"

                temp_text = task_per_span['text']
                # task_per_span['doc_id'] = task_per_span['doc_id']
                task_per_span['text'] = temp_text + str(task['doc_id']) + str(task['sentence_id'])
                # task_per_span = set_hashes(task_per_span)
                task_per_span['text'] = temp_text
                doc_id = task_per_span['doc_id'].replace('.xml', '.xml.txt')
                task_per_span['doc_host'] = f"http://{MY_IP}:{doc_port}/{doc_id}.html#{task_per_span['sentence_id']}"
                task_per_span['prop_holder'] = f"http://{MY_IP}:{pb_port}/{prop_holder}"

                for id_ in ['arg0', 'arg1', 'argL', 'argT']:
                    if id_ not in task_per_span or task_per_span[id_] == '':
                        task_per_span[id_] = ''
                        if curr_roleset and curr_roleset in roleset2args:
                            task_per_span[id_] = get_max_repeated_element(roleset2args[curr_roleset][id_])
                    pass

                task_per_span['answer'] = None

                yield task_per_span


def get_field_suggestions(dataset, propbank_dict):
    from prodigy.components.db import connect
    db = connect()
    dataset_arr = db.get_dataset(dataset)
    field_suggestions = {key: set() for key in [ROLESET_ID] + ALL_ARGS}
    field_suggestions['roleset_id'] = set(propbank_dict.keys())
    if dataset_arr is not None:
        for eg_ in dataset_arr:
            if eg_['answer'] == 'accept':
                for arg in ALL_ARGS:
                    if arg in eg_:
                        arg_val = eg_[arg]
                        if arg_val.replace('NA', '').strip() != '' and arg_val not in field_suggestions[arg]:
                            field_suggestions[arg].add(arg_val)
    return field_suggestions


def sort_tasks(stream):
    return sorted([t for t in stream],
                  key=lambda x: (x['doc_id'], int(x['sentence_id']), int(x['spans'][0]['start'])))


def get_rs_from_doc(doc, task_, topic_sense2vec):
    pass


@prodigy.recipe(
    "wsd-update",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file", "positional", None, str),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    port=("Port of the app", "option", 'port', int),
    pb_link=("Url to the PropBank website", 'option', 'pl', str)
)
def wsd_update(
    dataset: str,
    spacy_model: str,
    source: str,
    update: bool = False,
    port: Optional[int] = 8080,
    pb_link: Optional[str] = 'http://0.0.0.0:8701/',
):
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    labels = ['EVT']
    pb_dict = get_propbank_dict()

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Need jsonl file type for source.")

    stream = sort_tasks(stream)
    batch_size = 10

    topic_sense2vec = {}
    alias2roleset = defaultdict(set)

    for roleset, roledict in pb_dict.items():
        aliases = roledict['aliases']
        for alias in aliases:
            alias2roleset[alias].add(roleset)
    alias2roleset = {lemma: sorted(rolesets, key=lambda x: x.split('.')[-1]) for lemma, rolesets in
                     alias2roleset.items()}

    trs_arg2val_vec = {}
    field_suggestions = get_field_suggestions(dataset, pb_dict)

    def make_wsd_tasks2(stream_):
        texts = [(eg_["text"], eg_) for eg_ in stream_]
        for doc, task in nlp.pipe(texts, batch_size=batch_size, as_tuples=True):
            span = task['spans'][0]
            spacy_span = doc[span['token_start']: span['token_end'] + 1]
            span_root = spacy_span.root
            root_lemma = span_root.lemma_.lower()
            new_task = copy.deepcopy(task)
            new_task['roleset_id'] = ''

            if root_lemma in alias2roleset:
                possible_rs = alias2roleset[root_lemma]
            else:
                possible_rs = []

            topic = task['topic']
            new_task['prop_holder'] = pb_link
            new_task['field_suggestions'] = field_suggestions
            for arg_name in ALL_ARGS:
                new_task[arg_name] = ''
            if len(possible_rs):
                topic_rs = [(topic, rs) for rs in possible_rs]

                for trs in topic_rs:
                    if trs not in topic_sense2vec:
                        topic_sense2vec[trs] = np.ones(doc.vector.shape)

                cos_sims = [1 - cosine(topic_sense2vec[trs], doc.vector) for trs in topic_rs]
                # print(cos_sims)
                (_, best_roleset_id) = topic_rs[np.argmax(cos_sims)]

                curr_predicate = pb_dict[best_roleset_id]['frame']
                curr_roleset = pb_dict[best_roleset_id]['id']
                prop_holder = pb_link + '/' + curr_predicate + '.html#' + curr_roleset
                new_task[ROLESET_ID] = curr_roleset
                new_task['prop_holder'] = prop_holder

                # for arg_name in ALL_ARGS:
                #     trs_arg = (topic, best_roleset_id, arg_name)
                #     if trs_arg in trs_arg2val_vec:
                #         possible_args = trs_arg2val_vec[trs_arg].items()
                #         cos_sims_args = [1 - cosine(vec, doc.vector) for val, vec in possible_args]
                #         best_arg_val = possible_args[np.argmax(cos_sims_args)][0]
                #         new_task[arg_name] = best_arg_val

            new_task = set_hashes(new_task, input_keys=('text',), task_keys=('text', 'spans'), overwrite=True)
            yield new_task

    stream = make_wsd_tasks2(stream)

    def make_updates(answers):
        for answer in answers:
            if answer['answer'] == 'accept':
                if answer[ROLESET_ID]:
                    trs = (answer['topic'], answer[ROLESET_ID])
                    curr_roleid = trs[1]
                    # pb_dict[curr_roleid]['examples'].append(
                    #     [w for w in answer['text'].split() if w not in STOPWORDS])
                    if trs not in topic_sense2vec:
                        topic_sense2vec[trs] = nlp(answer['text']).vector
                    else:
                        topic_sense2vec[trs] = topic_sense2vec[trs] + nlp(answer['text']).vector
                    if answer['lemma'] not in alias2roleset:
                        alias2roleset[answer['lemma']] = [curr_roleid]
                    elif curr_roleid not in alias2roleset[answer['lemma']]:
                        alias2roleset[answer['lemma']].add(curr_roleid)

                    if curr_roleid not in field_suggestions[ROLESET_ID]:
                        field_suggestions[ROLESET_ID].append(curr_roleid)

    blocks = [
        {"view_id": "html", "html_template": PB_HTML},
        {"view_id": "html", "html_template": DOC_HTML2},
        {'view_id': 'ner'},
        {"view_id": "html", "html_template": HTML_INPUT, 'text': None},
        {'view_id': 'text_input', "field_rows": 1, "field_autofocus": False,
         "field_label": "Reason for Flagging"}
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        "auto_count_stream": not update,  # Whether to recount the stream at initialization
        "show_stats": True,
        "host": '0.0.0.0',
        "port": port,
        'blocks': blocks,
        'batch_size': batch_size,
        'history_length': batch_size,
        'instant_submit': False,
        "javascript": JAVASCRIPT_WSD + DO_THIS_JS_DISABLE
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_updates,
        "exclude": None,
        "config": config,
    }


@prodigy.recipe(
    "srl-update",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file", "positional", None, str),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    port=("Port of the app", "option", 'port', int),
    pb_link=("Url to the PropBank website", 'option', 'pl', str)
)
def srl_update(
    dataset: str,
    spacy_model: str,
    source: str,
    update: bool = False,
    port: Optional[int] = 8080,
    pb_link: Optional[str] = 'http://0.0.0.0:8701/',
):
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    labels = ['EVT']
    pb_dict = get_propbank_dict()

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Need jsonl file type for source.")

    stream = sort_tasks(stream)
    # print(stream[0]['answer'])
    batch_size = 10

    alias2roleset = defaultdict(set)

    for roleset, roledict in pb_dict.items():
        aliases = roledict['aliases']
        for alias in aliases:
            alias2roleset[alias].add(roleset)
    trs_arg2val_vec = {}
    field_suggestions = get_field_suggestions(dataset, pb_dict)
    topic_rs_arg2vals = {}

    def make_srl_tasks(stream_):
        texts = [(eg_["text"], eg_) for eg_ in stream_]
        for doc, task in nlp.pipe(texts, batch_size=batch_size, as_tuples=True):
            new_task = copy.deepcopy(task)
            topic = task['topic']
            new_task['field_suggestions'] = field_suggestions
            best_roleset_id = new_task[ROLESET_ID]
            curr_predicate = pb_dict[best_roleset_id]['frame']

            prop_holder = pb_link + '/' + curr_predicate + '.html#' + best_roleset_id

            new_task['prop_holder'] = prop_holder

            for arg_name in ALL_ARGS:
                trs_arg = (topic, best_roleset_id, arg_name)

                if trs_arg in topic_rs_arg2vals:
                    arg_vals = list(topic_rs_arg2vals[trs_arg])
                    possible_arg_vecs = [trs_arg2val_vec[trs_arg + (val,)] for val in arg_vals]
                    cos_sims_args = [1 - cosine(vec, doc.vector) for vec in possible_arg_vecs]
                    best_arg_val = arg_vals[np.argmax(cos_sims_args)]
                    new_task[arg_name] = best_arg_val
                else:
                    new_task[arg_name] = ''

            if 'answer' in new_task:
                new_task.pop('answer')
            new_task = set_hashes(new_task, input_keys=('text',), task_keys=('text', 'spans'), overwrite=True)
            yield new_task

    stream = make_srl_tasks(stream)

    def make_updates(answers):
        for answer in answers:
            if answer['answer'] == 'accept':
                best_roleset_id = answer[ROLESET_ID]
                topic = answer['topic']
                for arg_name in ALL_ARGS:
                    arg_val = answer[arg_name]
                    trs_arg = (topic, best_roleset_id, arg_name)
                    trs_arg_val = trs_arg + (arg_val,)
                    doc_vector = nlp(answer['text']).vector
                    if trs_arg_val in trs_arg2val_vec:
                        trs_arg2val_vec[trs_arg_val] = trs_arg2val_vec[trs_arg_val] + doc_vector
                    else:
                        trs_arg2val_vec[trs_arg_val] = doc_vector

                    if arg_val:
                        field_suggestions[arg_name].add(arg_val)
                        if trs_arg not in topic_rs_arg2vals:
                            topic_rs_arg2vals[trs_arg] = set()
                        topic_rs_arg2vals[trs_arg].add(arg_val)

    blocks = [
        {"view_id": "html", "html_template": PB_HTML},
        {"view_id": "html", "html_template": DOC_HTML2},
        {'view_id': 'ner'},
        {"view_id": "html", "html_template": HTML_INPUT, 'text': None},
        {'view_id': 'text_input', "field_rows": 1, "field_autofocus": False,
         "field_label": "Reason for Flagging"}
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        "auto_count_stream": not update,  # Whether to recount the stream at initialization
        "show_stats": True,
        "host": '0.0.0.0',
        "port": port,
        'blocks': blocks,
        'batch_size': batch_size,
        'history_length': batch_size,
        'instant_submit': False,
        "javascript": JAVASCRIPT_WSD + DO_THIS_JS
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_updates,
        "exclude": None,
        "config": config,
    }


@prodigy.recipe(
    "srl-manual",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file/ raw text or ltf directories", "positional", None, str),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    port=("Port of the app", "option", 'port', int),
    pb_link=("Url to the PropBank website", 'option', 'pl', str)
)
def srl_manual(
        dataset: str,
        spacy_model: str,
        source: str,
        update: bool = False,
        port: Optional[int] = 8080,
        pb_link: Optional[str] = 'http://0.0.0.0:8701/',
):
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    labels = ['EVT']

    pb_dict = pickle.load(open('./data/roleset.dict', 'rb'))

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Need jsonl file type for source.")

    stream_all = sorted([t for t in stream],
                        key=lambda x: (x['doc_id'], int(x['sentence_id']), int(x['spans'][0]['start'])))

    def make_stream(tasks):
        for t_ in tasks:
            t = copy.deepcopy(t_)
            t['field_suggestions'] = {key: set() for key in [ROLESET_ID] + ALL_ARGS}
            for key in [ROLESET_ID] + ALL_ARGS:
                t[key] = ''
            t['prop_holder'] = pb_link
            t['bert_doc'] = t['bert_doc'].replace(t['mention_id'], 'mark_id')
            t = set_hashes(t, input_keys=('text',), task_keys=('text', 'spans'), overwrite=True)
            yield t

    stream = make_stream(stream_all)
    batch_size = 5

    blocks = [
        # {"view_id": "html"},
        {"view_id": "html", "html_template": PB_HTML},
        {"view_id": "html", "html_template": DOC_HTML2},
        {'view_id': 'ner'},
        {"view_id": "html", "html_template": HTML_INPUT, 'text': None},
        {'view_id': 'text_input', "field_rows": 1, "field_autofocus": False,
         "field_label": "Reason for Flagging"}
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        "auto_count_stream": not update,  # Whether to recount the stream at initialization
        "show_stats": True,
        "host": '0.0.0.0',
        "port": port,
        'blocks': blocks,
        'batch_size': batch_size,
        'history_length': batch_size,
        'instant_submit': False,
        "javascript": JAVASCRIPT_WSD
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        # "update": make_updates,
        # "before_db": before_db,
        "exclude": None,
        "config": config,
        # "progress": progress,
        # "on_exit": on_exit,
        # "validate_answer": validate_answer,
        # "before_db": before_db
    }


@prodigy.recipe(
    "srl-fix",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file/ raw text or ltf directories", "positional", None, str),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    port=("Port of the app", "option", 'port', int),
    ann=("Annotator name", 'option', 'ann', str)
)
def srl_fix(
    dataset: str,
    spacy_model: str,
    source: str,
    update: bool = False,
    port: Optional[int] = 8080,
    ann: Optional[str] = '',
):
    """
    Review the wsd annotations
    """
    MY_IP = requests.request('GET', 'https://checkip.amazonaws.com/').text.strip()
    doc_port = 8700
    pb_port = 8701
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    labels = ['EVT']

    pb_dict = pickle.load(open('./data/roleset.dict', 'rb'))

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Need jsonl file type for source.")

    stream_all = sorted([t for t in stream],
                        key=lambda x: (x['doc_id'], int(x['sentence_id']), int(x['spans'][0]['start'])))

    for task in stream_all:
        task['doc_host'] = f"http://{MY_IP}:{doc_port}/{task['doc_id']}.txt.html#{task['sentence_id']}"
        if 'roleset_id' not in task:
            task['roleset_id'] = ''
        if task['roleset_id'] in pb_dict:
            task[
                'prop_holder'] = f"http://{MY_IP}:{pb_port}/{pb_dict[task['roleset_id']]['frame']}.html#{task['roleset_id']}"

        if 'annotator' not in task:
            task['annotator'] = ''
        if 'answer' in task:
            task.pop('answer')
        # task['_task_hash'] = hash(task['mention_id'])
        # task['_input_hash'] = -hash(task['mention_id'])

    stream = [task for task in stream_all if task['annotator'] == ann]

    if not len(stream):
        stream = stream_all
    num_total_tasks = len(stream)
    stream = add_tokens(nlp, stream)
    print('total tasks:', num_total_tasks)
    batch_size = 15

    field_suggestions = get_field_suggestions(dataset, pb_dict)
    fix_suggestions = {}

    def make_srl_fix_tasks(stream_):
        texts = [(eg_["text"], eg_) for eg_ in stream_]

        for doc, eg_ in nlp.pipe(texts, batch_size=batch_size, as_tuples=True):
            task_ = copy.deepcopy(eg_)
            task_['field_suggestions'] = field_suggestions
            roleset_id = task_['roleset_id']

            # task_['_input_hash'] = hash(task_['mention_id'])
            # task_['_task_hash'] = -hash(task_['mention_id'])
            yield task_

    def make_updates(answers):
        for answer in answers:
            roleset_id = answer['roleset_id']
            if roleset_id not in fix_suggestions:
                fix_suggestions[roleset_id] = {arg_: [] for arg_ in ALL_ARGS}
            for arg_ in ALL_ARGS:
                if answer[arg_].replace('NA', '').strip() != '':
                    fix_suggestions[roleset_id][arg_].append(answer[arg_])
                    field_suggestions[arg_].add(answer[arg_])

    stream = make_srl_fix_tasks(stream)

    def before_db(answers):
        print('before_db')
        without_suggestions = []
        for answer in answers:
            if 'field_suggestions' in answer:
                answer.pop('field_suggestions')
            without_suggestions.append(answer)
        return without_suggestions

    blocks = [
        # {"view_id": "html"},
        {"view_id": "html", "html_template": PB_HTML},
        {"view_id": "html", "html_template": DOC_HTML2},
        {'view_id': 'ner'},
        {"view_id": "html", "html_template": HTML_INPUT, 'text': None},
        {'view_id': 'text_input', "field_rows": 1, "field_autofocus": False,
         "field_label": "Reason for Flagging"}
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        "auto_count_stream": not update,  # Whether to recount the stream at initialization
        "show_stats": True,
        "host": '0.0.0.0',
        "port": port,
        'blocks': blocks,
        'batch_size': batch_size,
        'history_length': batch_size,
        'instant_submit': False,
        "javascript": JAVASCRIPT_WSD
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_updates,
        "before_db": before_db,
        "exclude": None,
        "config": config,
        # "progress": progress,
        # "on_exit": on_exit,
        # "validate_answer": validate_answer,
        # "before_db": before_db
    }


@prodigy.recipe(
    "srl-review",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file/ raw text or ltf directories", "positional", None, str),
    port=("Port of the app", "option", 'port', int),
    ann=("Annotator name", 'option', 'ann', str)
)
def srl_review(
    dataset: str,
    spacy_model: str,
    source: str,
    port: Optional[int] = 8080,
    ann: Optional[str] = '',
):
    """
    Review the wsd annotations
    """
    MY_IP = requests.request('GET', 'https://checkip.amazonaws.com/').text.strip()
    doc_port = 8700
    pb_port = 8701
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    labels = ['EVT']

    pb_dict = pickle.load(open('./data/roleset.dict', 'rb'))

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Need jsonl file type for source.")

    stream_all = sorted([t for t in stream],
                        key=lambda x: (x['doc_id'], int(x['sentence_id']), int(x['spans'][0]['start'])))

    for task in stream_all:
        task['doc_host'] = f"http://{MY_IP}:{doc_port}/{task['doc_id']}.txt.html#{task['sentence_id']}"
        if 'roleset_id' not in task:
            task['roleset_id'] = ''
        if task['roleset_id'] in pb_dict:
            task[
                'prop_holder'] = f"http://{MY_IP}:{pb_port}/{pb_dict[task['roleset_id']]['frame']}.html#{task['roleset_id']}"

        if 'annotator' not in task:
            task['annotator'] = ''
        # task['_task_hash'] = hash(task['mention_id'])
        # task['_input_hash'] = -hash(task['mention_id'])

    stream = [task for task in stream_all if task['annotator'] == ann]

    if not len(stream):
        stream = stream_all
    num_total_tasks = len(stream)
    stream = add_tokens(nlp, stream)
    print('total tasks:', num_total_tasks)
    batch_size = 15

    field_suggestions = get_field_suggestions(dataset, pb_dict)
    fix_suggestions = {}

    def make_srl_fix_tasks(stream_):
        texts = [(eg_["text"], eg_) for eg_ in stream_]

        for doc, eg_ in nlp.pipe(texts, batch_size=batch_size, as_tuples=True):
            task_ = copy.deepcopy(eg_)
            task_['field_suggestions'] = field_suggestions
            roleset_id = task_['roleset_id']
            # new_task = set_hashes(task_)
            # task_['_input_hash'] = hash(task_['mention_id'])
            # task_['_task_hash'] = -hash(task_['mention_id'])
            yield task_

    stream = make_srl_fix_tasks(stream)

    def before_db(answers):
        print('before_db')
        without_suggestions = []
        for answer in answers:
            if 'field_suggestions' in answer:
                answer.pop('field_suggestions')
            without_suggestions.append(answer)
        return without_suggestions

    blocks = [
        # {"view_id": "html"},
        {"view_id": "html", "html_template": PB_HTML},
        {"view_id": "html", "html_template": DOC_HTML},
        {'view_id': 'ner'},
        {"view_id": "html", "html_template": HTML_INPUT, 'text': None},
        {'view_id': 'text_input', "field_rows": 1, "field_autofocus": False,
         "field_label": "Reason for Flagging"}
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options

        "show_stats": True,
        "host": '0.0.0.0',
        "port": port,
        'blocks': blocks,
        'batch_size': batch_size,
        'history_length': batch_size,
        'instant_submit': False,
        "javascript": JAVASCRIPT_WSD
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "before_db": before_db,
        "exclude": None,
        "config": config,
        # "progress": progress,
        # "on_exit": on_exit,
        # "validate_answer": validate_answer,
        # "before_db": before_db
    }


class TestSRL(unittest.TestCase):
    def test_srl_update_new_tasks(self):
        source = './data/dev/dev_george.jsonl'
        spacy_model = 'en_core_web_md'
        datas = 'test_dataset'
        ctlr = srl_update(datas, spacy_model, source, update=True)
        cltr_stream = ctlr['stream']
        update_func = ctlr['update']
        for task in cltr_stream:
            assert 'answer' not in task

    def test_srl_update(self):
        source = './data/dev/dev_george.jsonl'
        spacy_model = 'en_core_web_md'
        datas = 'test_dataset'
        ctlr = srl_update(datas, spacy_model, source, update=True)
        cltr_stream = ctlr['stream']
        update_func = ctlr['update']
        for task in cltr_stream:
            for arg_n in ALL_ARGS:
                task[arg_n] = arg_n + str(random.randint(0, 3))
            task['answer'] = 'accept'
            update_func([task])


if __name__ == '__main__':
    unittest.main()