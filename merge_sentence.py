
import json
import matplotlib.pyplot as plt
import networkx as nx
from utils2 import merge_nodes, overlap
import glob, os
from tqdm import tqdm
from copy import deepcopy
from data_stats import extract_bayesian_net, extract_bayesian_net_knn, knn_with_type
from pprint import pprint
import pickle as pkl


def label_smoothing(rel, labels, dist, e=0.2):
    for i, label in enumerate(labels):
        if rel == label:
            delta = 1
        else:
            delta = 0
        dist[i] = delta * (1-e) + e * dist[i]
    return dist


def compute_dist(sdp, all_sdps, all_rels):
    dist = {}
    for key in set(all_rels):
        dist[key] = 0
    for _sdp, _rel in zip(all_sdps, all_rels):
        if sdp == _sdp:
            dist[_rel] += 1
    total = sum(dist[key] for key in dist)
    dist_ratio = deepcopy(dist)
    for key in dist:
        dist_ratio[key] = dist[key] / total
    return dist, dist_ratio


def format_dist(net, nbrs, sdp, rels, equal=False):
    if sdp in net:
        net = net[sdp]
        res = []
        count = 0
        for rel in rels:
            if rel in net:
                res.append(net[rel][0] / 100)
                count += 1
            else:
                res.append(0)
        if equal:
            res = [1.0/count if i != 0 else 0.0 for i in res]
        return res
    else:
        res = nbrs.predict([sdp])[0]
        total = sum(v for k, v in res.items())
        value = {k: v / total for k, v in res.items()}
        value = [res[rel] if rel in value else 0.0 for rel in rels]
        return value


def create_distribution(bayesian_net,
                        nbrs,
                        length,
                        data_type='train',
                        data_path='dataset/tacred/{}.json'):
    with open(data_path.format(data_type), 'r') as f0:
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
        docs = json.load(f0)
        all_sdps = []
        all_rels = []
        for doc in tqdm(docs):
            sdp = merge_sentence(doc)
            doc['sdp'] = sdp
            # all_sdps.append(sdp)
            # all_rels.append(doc['relation'])
        for doc in tqdm(docs):
            # dist_raw, dist_ratio = compute_dist(doc['sdp'], all_sdps, all_rels)
            # doc['dist_raw'] = [dist_raw.get(key, 0) for key in allowed_relations]
            # doc['dist_ratio'] = [dist_ratio.get(key, 0) for key in allowed_relations]
            # doc['dist_smooth'] = label_smoothing(doc['relation'], allowed_relations, doc['dist_ratio'])
            sdp = doc['sdp']
            if length != -1:
                sdp = tuple([sdp[0]] + list(sdp[1:1 + length]) + [sdp[-1]])
            else:
                sdp = tuple(sdp)
            doc['dist_length'] = format_dist(bayesian_net, nbrs, sdp, allowed_relations)
            doc['dist_equal'] = format_dist(bayesian_net, nbrs, sdp, allowed_relations, True)
            if 'dist_smooth' in doc:
                del doc['dist_smooth']
            if 'dist_smooth_len' in doc:
                del doc['dist_smooth_len']
            if 'dist_smooth_equal' in doc:
                del doc['dist_smooth_equal']
            #doc['dist_smooth'] = label_smoothing(doc['relation'], allowed_relations, doc['dist_ratio'])
            #doc['dist_smooth_len'] = label_smoothing(doc['relation'], allowed_relations, doc['dist_length'])
            #doc['dist_smooth_equal'] = label_smoothing(doc['relation'], allowed_relations, doc['dist_equal'])
    with open(data_path.format(data_type), 'w') as f0:
        json.dump(docs, f0)


def merge_sentence(doc):
    pos_tags = doc['stanford_pos']
    nes = doc['stanford_ner']
    deps = doc['stanford_head']
    head = {'start': doc['subj_start'], 'end': doc['subj_end'], 'type': doc['subj_type']}
    tail = {'start': doc['obj_start'], 'end': doc['obj_end'], 'type': doc['obj_type']}
    deps = [x - 1 for x in deps]
    # Build a graph
    graph = nx.Graph()
    for i, tag in enumerate(pos_tags):
        if deps[i] == -1:
            continue
        # print(i, tag, deps[i])
        edge = (
                '{0}-{1}'.format(deps[i], pos_tags[deps[i]]),
                '{0}-{1}'.format(i, tag),
        )
        graph.add_edge(*edge)

    chunks = []
    i = 0
    while i < len(pos_tags) - 1:
        tag = [pos_tags[i]]
        ne = [nes[i]]
        start = i
        end = i
        if ne[-1] == 'O':
            chunks.append((tag, ne, start, end))
            i += 1
            continue
        else:
            j = i + 1
            while j < len(pos_tags):
                if nes[j] != ne[-1]:
                    chunks.append((tag, ne, start, end))
                    # i = j
                    break
                else:
                    tag.append(pos_tags[j])
                    ne.append(nes[j])
                    end += 1
                    j += 1
            i = j
    named_chunks = []
    for chunk in chunks:
        tag, ne, start, end = chunk
        # print('tag, ne, start, end: ', tag, ne, start, end)
        if ne.count(ne[0]) != len(ne):
            print(doc['docid'])
        assert ne.count(ne[0]) == len(ne)
        if ne[0] != 'O':
            if overlap(start, end, head['start'], head['end']) or \
                    overlap(start, end, tail['start'], tail['end']):
                continue
            named_chunks.append((start, end, ne[0]))
    named_chunks.append((head['start'], head['end'], head['type']))
    named_chunks.append((tail['start'], tail['end'], tail['type']))

    for chunk in named_chunks:
        start, end, wtype = chunk
        # print('start, end, wtype: ', start, end, wtype)
        nodes = ['{0}-{1}'.format(j, pos_tags[j]) for j in range(start, end+1)]
        # print('nodes: ', nodes)
        new_node = '{0}-{1}'.format(start, wtype)
        # print('new_node: ', new_node)
        try:
            merge_nodes(graph, nodes, new_node)
        except nx.exception.NetworkXError as e:
            print(doc['docid'])
            print(e)
    # for node in graph.nodes(data=True):
    #     print(node)
    # print('=====================')
    # for edge in graph.edges(data=True):
    #     print(edge)
    try:
        head_word = '{0}-{1}'.format(head['start'], head['type'])
        tail_word = '{0}-{1}'.format(tail['start'], tail['type'])
        shortest_dep_path = nx.shortest_path(graph,
                                             source=head_word,
                                             target=tail_word)
    except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound) as e:
        print(e)
        print(doc['docid'])
        return [head['type'], tail['type']]
    return [path.split('-')[1] for path in shortest_dep_path]


if __name__ == '__main__':
    # bayesian_net, nbrs = extract_bayesian_net_knn(-1, k=10)
    bayesian_net = extract_bayesian_net(-1, test_included=True)
    nbrs = None
    # with open('dataset/tacred/bayesian_net_max_dist_30.pkl', 'wb') as f:
    #     pkl.dump(bayesian_net, f)
    # with open('dataset/tacred/nbrs_max_dist_30.pkl', 'wb') as f:
    #     pkl.dump(nbrs, f)
    # with open('dataset/tacred/bayesian_net_full_sdp_k40.pkl', 'rb') as f:
    #     bayesian_net = pkl.load(f)
    # with open('dataset/tacred/nbrs_full_sdp_k40.pkl', 'rb') as f:
    #     nbrs = pkl.load(f)
    # pprint(bayesian_net)

    # print('train')
    # create_distribution(bayesian_net, nbrs, -1, 'train')
    # print('dev')
    # create_distribution(bayesian_net, nbrs, -1, 'dev')
    print('test')
    create_distribution(bayesian_net, nbrs, -1, 'test')





