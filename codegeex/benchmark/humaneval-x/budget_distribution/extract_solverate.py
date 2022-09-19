# This file is for gathering the solve rates from generated files
import json
import os

import numpy as np

language = ['cpp', 'java', 'js', 'python', 'go']
repo = "<directory_to_generated_jsonl_files>"

all_reps = os.listdir(repo)

# choose the ones
all_passes = {}
assignment = [33, 6, 20, 32, 9]
assignment = [38, 8, 29, 17, 8]
assignment = [12, 4, 5, 45, 34]
for folder in all_reps:
    if not ("." in folder):
        q = os.listdir(repo + '/' + folder)
        for f in q:
            if 'result' in f and not ('example' in f):
                passed = np.zeros(164)
                all_p = 0
                fi = open(repo + '/' + folder + '/' + f, 'r')
                t = fi.readlines()
                for l in t:
                    if len(l.strip()) == 0:
                        continue
                    qq = json.loads(l)
                    if qq['passed'] == True:
                        id = int(qq['task_id'].split('/')[1])
                        passed[id] += 1
                        all_p += 1
                all_passes[f] = list(passed)
                print(f, all_p)

json.dump(all_passes, open('solve_rate_final.jsonl', 'w'))
