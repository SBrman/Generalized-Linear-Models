#! python3

import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, xlabel, ylabel, title, d, ddd=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y, c='g' if d else 'r', label='yes' if d else 'no')
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel.format('yes' if d else 'no'), fontsize=13)
    ax.set_title(title.format('yes' if d else 'no', xlabel, ddd))
    plt.savefig(f'{xlabel}{ylabel}{d}.png')

data = """1 1 65 317 2 20
1 2 65 476 5 33
1 3 52 486 4 40
1 4 310 3259 36 316
2 1 98 486 7 31
2 2 159 1004 10 81
2 3 175 1355 22 122
2 4 877 7660 102 724
3 1 41 223 5 18
3 2 117 539 7 39
3 3 137 697 16 68
3 4 477 3442 63 344
4 1 11 40 0 3
4 2 35 148 6 16
4 3 39 214 8 25
4 4 167 1019 33 114"""

data = np.array([[int(i) for i in line.split(' ')]
                 for line in data.split('\n')])

car = data[:, 0]
age = data[:, 1]
dist0 = data[:, 2:4]
dist1 = data[:, 4:]

def plot(kv, xlabel, ylabel, title):
    xy = {k: v['yes']/(v['yes'] + v['no']) for k, v in kv.items()}
    fig, ax = plt.subplots()
    plt.scatter(xy.keys(), xy.values())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 5, 2)
    plt.show()

def dictmaker(zipped_kv):
    cd = {}
    for cc, (yy, nn) in zipped_kv:
        cd.setdefault(cc, {})
        cd[cc].setdefault("yes", 0)
        cd[cc].setdefault("no", 0)
        cd[cc]['yes'] += yy
        cd[cc]['no'] += nn

    return cd

d = []
for dd in [dist0, dist1]:
    for d1, d2 in dd:
        d.append([d1, d2])

##c = np.append(car, car)
##cdd = list(zip(c, d))
##cd = dictmaker(cdd)
##plot(cd, 'CAR', 'Rate of claims', 'Rate of claims vs CAR type')
##
##a = np.append(age, age)
##add = list(zip(a, d))
##ad = dictmaker(add)
##plot(ad, 'AGE', 'Rate of claims', 'Rate of claims vs Age')

##dy0, dn0 = sum(dist0)
##dist0_claim_rate = dy0 / (dy0 + dn0)
##
##dy1, dn1 = sum(dist1)
##dist1_claim_rate = dy1 / (dy1 + dn1)
##
##fig, ax = plt.subplots()
##plt.bar([0, 1], [dist0_claim_rate, dist1_claim_rate])
##ax.set_ylim(0.10, 0.15)
##plt.xticks([0, 1])
##ax.set_xlabel('DIST')
##ax.set_ylabel('Rate of Claims')
##ax.set_title('Rate of claims vs DIST')
####ax.set_xlim(-0.5, 1.5, 1)
##plt.show()
