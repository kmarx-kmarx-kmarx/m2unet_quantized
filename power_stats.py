import re
path = "./power_stats.txt"
save = './power_csv.csv'

with open(path) as f:
    lines = f.readlines()

t0 = -1

with open(save, 'w') as f:
    for line in lines:
        sp = line.split(' ')
        cpu_info = sp[11]
        stats = re.findall(r'\d+', cpu_info)
        fom = 0
        for i in range(int(len(stats)/2)):
            idx = i * 2
            fom += float(stats[idx])*float(stats[idx+1])
        t = sp[1]
        t = t.split(':')
        ti = int(t[-1]) + 60*int(t[-2]) + 60*60*int(t[-3])
        if t0 == -1:
            t0 = ti
        f.write(f'{ti-t0},{fom}\n')