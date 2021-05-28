import re
import pickle


for i in range(5, 15):
    f = "history" + str(i)

    with open(f, 'r') as fp:
        x = fp.readlines()

    regex = re.compile(r'\d+\.\d+')

    history = {'mse': [], 'val_mse': [], 'p*_mse': [], 'loss': [],
               'val_loss': [], 'p*_loss': [], 'p*_acc': [], 'acc': [],
               'val_acc':[]}

    for line in x:
        if line[:9] == "1563/1563":
            vals = regex.findall(line)
            vals = [float(x) for x in vals]
            history['loss'] = history['loss'] + [vals[0]]
            history['acc'] = history['acc'] + [vals[1]]
            history['mse'] = history['mse'] + [vals[2]]
            history['val_loss'] = history['val_loss'] + [vals[3]]
            history['val_acc'] = history['val_acc'] + [vals[4]]
            history['val_mse'] = history['val_mse'] + [vals[5]]
        elif line[:7] == "p*_loss":
            history['p*_loss'] = history['p*_loss'] + [float(regex.findall(line)[0])]
        elif line[:6] == "p*_acc":
            history['p*_acc'] = history['p*_acc'] + [float(regex.findall(line)[0])]
        elif line[:6] == "p*_mse":
            history['p*_mse'] = history['p*_mse'] + [float(regex.findall(line)[0])]

    with open(f + "_new.pkl", "wb") as fp:
        pickle.dump(history, fp, pickle.HIGHEST_PROTOCOL)
