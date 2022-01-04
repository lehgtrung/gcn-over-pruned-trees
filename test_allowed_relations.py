
import json
from sklearn.metrics import classification_report, f1_score


if __name__ == '__main__':
    with open('dataset/tacred/test.json', 'r') as f:
        docs = json.load(f)
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
    gts = []
    preds = []
    for doc in docs:
        dist = doc['dist_length']
        gt = doc['relation']
        zipper = dict(zip(allowed_relations, dist))
        pred = max(zipper, key=zipper.get)
        # print('gt: {}, pred: {}'.format(gt, pred))
        gts.append(gt)
        preds.append(pred)
    print(classification_report(gts, preds, labels=allowed_relations))
    print(f1_score(gts, preds, average='micro'))
    print(f1_score(gts, preds, average='macro'))




