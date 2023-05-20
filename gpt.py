import re
import lexnlp
import pickle

instructions_prompt = """You are a concise annotator that follows these instructions:
1. Identify the target event trigger lemma and its correct roleset sense in the given text.
2. Annotate the document-level ARG-0 and ARG-1 roles using the PropBank website for the roleset definitions.
3. If the ARG-1 role is an event, identify the head predicate and provide its roleset ID.
4. Perform within-document and cross-document anaphora resolution of the ARG-0 and ARG-1 using Wikipedia.
5. Use external resources, such as Wikipedia, to annotate ARG-Loc and ARG-Time."""

standard_event_eg = """sentence: HP today announced that it has signed a definitive agreement to <m> acquire </m> EYP Mission Critical Facilities Inc.
your response:
roleset : acquire.01
ARG-0 :HP (referencing https://en.wikipedia.org/wiki/Hewlett-Packard)
ARG-1 : EYP
ARG-Loc : https://en.wikipedia.org/wiki/United_States
ARG-Time : November 12, 2007"""

eventive_event_eg = """sentence with eventive argument (ARG-1 is a roleset): HP today announced that it has <m> signed </m> a definitive agreement to acquire EYP Mission Critical Facilities Inc
your response:
roleset : sign.02
ARG-0 : it (referencing https://en.wikipedia.org/wiki/Hewlett-Packard)
ARG-1 : agree.01
ARG-Loc : https://en.wikipedia.org/wiki/United_States
ARG-Time : November 12, 2007"""

print('\n'.join([instructions_prompt, standard_event_eg, eventive_event_eg]))


class Argument:
    def __init__(self, line):
        self.arg_name = line.split(':')[0]
        arg_value = ''.join([s.strip() for s in line.split(':')[1:]])
        self.parse_arg_value(arg_value)
        self.arg_text = ''
        self.arg_reference = ''
        self.wiki = None
        self.non_wiki = None
        self.rolesets = None
        self.dates = None
        self.args_multi = None

    def parse_arg_value(self, text):
        self.arg_text = text.split('(')[0].strip()
        self.arg_reference = text.split('(')[-1].strip(')').replace('referencing', '').strip()

        self.wiki = [w.split('wiki/')[-1].split('#')[0] for w in re.findall(r'(wiki/\S+)', self.arg_reference)]
        self.non_wiki = [s.replace('https//', '').replace('http//', '').replace('www.', '').strip('.com')
                         for s in re.findall(r'(https?:?//\S+)', self.arg_reference) if 'wikipedia' not in s]
        self.rolesets = re.findall(r'.*\.\d{1,3}', text)

        self.dates = list(lexnlp.extract.en.dates.get_dates(text))
        self.args_multi = [s.strip() for s in self.arg_text.split('and')]

    def __str__(self):
        return ' '.join([f'[{self.arg_name}]', f'[{self.arg_text}]', f'[args: {self.args_multi}]'
                         f'[dates: {self.dates}]'
                         f'[rolesets: {self.rolesets}]',
                         f'[non-wiki: {self.non_wiki}]',
                         f'[wiki: {self.wiki}]', f'[{self.arg_reference}]'])