import json
from tqdm import tqdm
from pprint import pprint
from math import floor, ceil
from collections import Counter
import pickle as pkl
from utils2 import levenshtein
import random


class KNeighborsClassifier:
    def __init__(self, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.bucket = None
        self.cache = {}

    def fit(self, bucket):
        self.bucket = bucket

    def predict(self, X):
        values = []
        for x in X:
            dist = []
            base_sdp = (x[0], x[-1])
            if x in self.cache:
                values.append(self.cache[x])
                break
            for _x, _y in zip(self.bucket[base_sdp]['X'], self.bucket[base_sdp]['y']):
                metric = self.metric(_x, x)
                # metric = metric / max(len(x), len(_x))  # normalize
                if self.n_neighbors == 1 and metric == 0 and isinstance(_y, str):
                    dist.append((metric, _y))
                    break
                if isinstance(_y, str):
                    dist.append((metric, _y))
            sorted_dist = sorted(dist, key=lambda c: c[0])
            sorted_dist = [e[1] for e in sorted_dist[:self.n_neighbors]]
            value = Counter(sorted_dist)
            # total = sum(v for k, v in value.items())
            # value = {k: v/total for k, v in value.items()}
            values.append(value)
            self.cache[x] = value
        return values


def extract_bayesian_net(length, unique=False, threshold=3, test_included=False):
    docs = []
    with open('dataset/tacred/dev.json', 'r') as f:
        docs.extend(json.load(f))
    if test_included:
        with open('dataset/tacred/dev.json', 'r') as f:
            docs.extend(json.load(f))
        with open('dataset/tacred/test.json', 'r') as f:
            docs.extend(json.load(f))
    net = {}
    sdp_net = {}
    for doc in tqdm(docs):
        rel = doc['relation']
        # if rel == 'no_relation':
        #     continue
        subj = doc['subj_type']
        obj = doc['obj_type']
        if 'sdp' in doc:
            sdp = tuple(doc['sdp'])
            if length != -1:
                sdp = tuple([sdp[0]] + list(sdp[1:1 + length]) + [sdp[-1]])
            else:
                sdp = tuple(sdp)
        else:
            sdp = (subj, obj)
        if subj not in net:
            net[subj] = {}
        else:
            if obj not in net[subj]:
                net[subj][obj] = {}
            if sdp not in net[subj][obj]:
                net[subj][obj][sdp] = {}
            if rel not in net[subj][obj][sdp]:
                net[subj][obj][sdp][rel] = 1
            else:
                net[subj][obj][sdp][rel] += 1

    for _type in net:
        for _atype in net[_type]:
            for _sdp in net[_type][_atype]:
                total = sum(net[_type][_atype][_sdp][_rel] for _rel in net[_type][_atype][_sdp])
                for _rel in net[_type][_atype][_sdp]:
                    net[_type][_atype][_sdp][_rel] = (
                        float('{:.3f}'.format(net[_type][_atype][_sdp][_rel] / total * 100)),
                        net[_type][_atype][_sdp][_rel]
                    )
                    sdp_net[_sdp] = net[_type][_atype][_sdp]
    if unique:
        sdp_net = {k: v for k, v in sdp_net.items() if len(v.keys()) == 1 and v[list(v.keys())[0]][1] >= threshold}
    return sdp_net


def knn_with_type(docs, k):
    X = y = []
    bucket = {}
    for doc in tqdm(docs):
        base_sdp = (doc['sdp'][0], doc['sdp'][-1])
        if base_sdp not in bucket:
            bucket[base_sdp] = {}
            bucket[base_sdp]['X'] = []
            bucket[base_sdp]['y'] = []
        bucket[base_sdp]['X'].append(doc['sdp'])
        bucket[base_sdp]['y'].append(doc['relation'])

    nbrs = KNeighborsClassifier(n_neighbors=k, metric=levenshtein)
    nbrs.fit(bucket)
    return nbrs


def extract_bayesian_net_knn(length, k=5):
    docs = []
    with open('dataset/tacred/train.json', 'r') as f:
        docs.extend(json.load(f))
    net = {}
    sdp_net = {}
    nbrs = knn_with_type(docs, k=k)
    for doc in tqdm(docs):
        subj = doc['subj_type']
        obj = doc['obj_type']
        if 'sdp' in doc:
            sdp = tuple(doc['sdp'])
            if length != -1:
                sdp = tuple([sdp[0]] + list(sdp[1:1 + length]) + [sdp[-1]])
            else:
                sdp = tuple(sdp)
        else:
            sdp = (subj, obj)
        if not subj in net:
            net[subj] = {}
        if not obj in net[subj]:
            net[subj][obj] = {}
        if sdp not in net[subj][obj]:
            net[subj][obj][sdp] = nbrs.predict([sdp])[0]
    for _type in net:
        for _atype in net[_type]:
            for _sdp in net[_type][_atype]:
                total = sum(net[_type][_atype][_sdp][_rel] for _rel in net[_type][_atype][_sdp])
                for _rel in net[_type][_atype][_sdp]:
                    net[_type][_atype][_sdp][_rel] = (
                        float('{:.3f}'.format(net[_type][_atype][_sdp][_rel] / total * 100)),
                        net[_type][_atype][_sdp][_rel]
                    )
                    sdp_net[_sdp] = net[_type][_atype][_sdp]
    return sdp_net, nbrs


def extract_sdp():
    with open('dataset/tacred/train.json', 'r') as f:
        docs = json.load(f)
        net = {}
        allowed_relations = ['no_relation',
                             'per:title',
                             'org:top_members/employees',
                             'per:employee_of',
                             'org:alternate_names',
                             'org:country_of_headquarters',
                             'per:countries_of_residence',
                             'org:city_of_headquarters',
                             'per:cities_of_residence',
                             'per:age',
                             'per:stateorprovinces_of_residence',
                             'per:origin',
                             'org:subsidiaries',
                             'org:parents',
                             'per:spouse',
                             'org:stateorprovince_of_headquarters',
                             'per:children',
                             'per:other_family',
                             'per:alternate_names',
                             'org:members',
                             'per:siblings',
                             'per:schools_attended',
                             'per:parents',
                             'per:date_of_death',
                             'org:member_of',
                             'org:founded_by',
                             'org:website',
                             'per:cause_of_death',
                             'org:political/religious_affiliation',
                             'org:founded',
                             'per:city_of_death',
                             'org:shareholders',
                             'org:number_of_employees/members',
                             'per:date_of_birth',
                             'per:city_of_birth',
                             'per:charges',
                             'per:stateorprovince_of_death',
                             'per:religion',
                             'per:stateorprovince_of_birth',
                             'per:country_of_birth',
                             'org:dissolved',
                             'per:country_of_death']
        for rel in allowed_relations:
            net[rel] = []
        for doc in docs:
            # if doc['relation'] == 'no_relation':
            #     prob = random.uniform(0, 1)
            #     if prob > 0.3:
            #         continue
            net[doc['relation']].append(doc['sdp'])
        with open('dataset/tacred/sdps.json', 'w') as f:
            json.dump(net, f)


if __name__ == '__main__':
    net = extract_bayesian_net(-1, unique=False, threshold=1)
    pprint(net)
    # extract_sdp()





