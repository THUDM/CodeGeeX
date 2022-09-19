# This file is for evaluating the budget distribution method
import json
import numpy as np

w = json.load(open("solve_rate_final.jsonl", 'r'))


def build_chart():
    fa = np.ones((201, 201))

    for i in range(1, 201):
        for j in range(201):
            fa[j, i] = fa[j, i - 1] * (201 - j - i) / (201 - i)

    return fa


languages = ['cpp', 'go', 'java', 'python', 'js']
models = ['codegeex', 'codegen16b', 'codegen6b', 'incoder']
fa = build_chart()


def compute(l, dist):
    budgets = []
    alldists = []
    for i in range(2, 41):
        budgets.append(i * 5)
        alldists.append(distribute(dist, i * 5))
    # print(alldists)
    sums = np.zeros(39)
    sumdists = np.zeros(39)
    sumop = np.zeros((39, 5))
    summax = np.zeros(39)
    for i in range(164):
        currents = np.ones(39)
        currentdists = np.ones(39)
        currentops = np.ones((39, 5))

        for w in range(5):
            num = int(l[w][i])
            for j in range(39):
                currents[j] = currents[j] * fa[j + 2, num]

                currentdists[j] = currentdists[j] * fa[alldists[j][w], num]

                currentops[j, w] = fa[(j + 2) * 5, num]

        sums = sums + (1 - currents)
        sumdists = sumdists + (1 - currentdists)
        sumop = sumop + (1 - currentops)
        summax = summax + (1 - np.min(currentops, axis=1))

    sumop = np.max(sumop, axis=1)
    return sums / 164, sumdists / 164, sumop / 164, summax / 164


def distribute(distribution, budget):
    sum = np.sum(distribution)
    di = np.array(distribution) / sum * budget
    dis = []
    diff = []
    for i in range(len(di)):
        dis.append(int(di[i]))
        diff.append(dis[i] - di[i])
    # overflow assignment
    need = np.sum(dis) - budget
    while need > 0:
        g = np.argmax(diff)
        dis[g] -= 1
        diff[g] -= 1
        need -= 1
    while need < 0:
        g = np.argmin(diff)
        dis[g] += 1
        diff[g] += 1
        need += 1
    return dis


names = []
for i in range(39):
    names.append(str((i + 2) * 5) + " uniform")
for i in range(39):
    names.append(str((i + 2) * 5) + " weighted")
for i in range(39):
    names.append(str((i + 2) * 5) + " best")
for i in range(39):
    names.append(str((i + 2) * 5) + " max")

out = open("solution_output.txt", 'w')
for model in models:
    if 'codegeex' in model:
        dist = [33, 6, 20, 32, 9]
    if 'codegen' in model:
        dist = [38, 8, 29, 17, 8]
    if 'incoder' in model:
        dist = [12, 4, 5, 45, 34]
    avi_list = {}
    for pp in w:
        if (np.sum(w[pp]) > 1500):
            if model in pp:
                for l in languages:
                    if l in pp.replace('javascript', 'js'):
                        if l in avi_list:
                            avi_list[l].append(pp)
                        else:
                            avi_list[l] = [pp]
    # print(avi_list)
    maxsums = np.zeros(len(names))
    maxsumscomb = np.zeros((len(names), 5))
    current_marker = [0, 0, 0, 0, 0]
    while current_marker[0] < len(avi_list[languages[0]]):
        aclist = []
        for i in range(5):
            aclist.append(w[avi_list[languages[i]][current_marker[i]]])
        sums, sumdists, sumop, summax = compute(aclist, dist)
        things = np.concatenate((sums, sumdists, sumop, summax))
        for i in range(len(names)):
            if (things[i] > maxsums[i]):
                # print(names[i],things[i],current_marker)
                maxsums[i] = things[i]
                maxsumscomb[i] = current_marker

        current_marker[-1] += 1
        p = 4
        while (current_marker[p] >= len(avi_list[languages[p]]) and p > 0):
            current_marker[p] = 0
            current_marker[p - 1] += 1
            p -= 1

    print(model)
    print(model, file=out)
    for i in range(len(names)):
        print(names[i], maxsums[i], maxsumscomb[i])
        print(names[i], maxsums[i], file=out)
    # use the best of mix100 for further purposes
    for i in range(5):
        print(languages[i], avi_list[languages[i]][int(maxsumscomb[2, i])])
out.close()
