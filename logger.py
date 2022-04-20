import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

import torch


def classification_report_df(report):
    pattern = re.compile(r'(?P<Class> \d\.\d)'
                         r'\s+(?P<Precision>\d\.\d{2})'
                         r'\s+(?P<recall>\d\.\d{2})'
                         r'\s+(?P<F1_score>\d\.\d{2})'
                         r'\s+(?P<Support>\d+)')
    report_data = ','.join(pattern.groupindex.keys()) + '\n'

    lines = report.split('\n')
    for line in lines[2:-3]:
        row_data = pattern.findall(line)
        if row_data:
            report_data += ','.join(row_data[0]) + '\n'
    for line in lines[-4:]:
        line = line.strip()
        new = ''
        for i in range(len(line)):
            if line[i] == ' ' and line[i-1].isalpha() and line[i+1].isalpha():
                new += '_'
            elif not (line[i] == ' ' and line[i-1] == ' '):
                new += line[i]
        new = new.replace(' ', ',')
        report_data += new + '\n'

    return report_data


def plot_train_test(loss_train, acc_train, loss_test, acc_test):
    epochs = np.arange(len(loss_train))
    figs = []
    for i, d in zip(range(1, 3), ([loss_train, loss_test, 'loss'], [acc_train, acc_test, 'accuracy'])):
        plt.figure(i)
        plt.plot(epochs, d[0], label=f'train {d[2]}')
        plt.plot(epochs, d[1], label=f'test {d[2]}')
        plt.xlabel('Epochs')
        plt.ylabel(d[2].capitalize())
        plt.xticks(epochs, labels=epochs, rotation=90)
        plt.legend()
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
        plt.tight_layout()
        figs.append(plt.gcf())
    return figs


def log(model, bs, lr, optimizer, epochs, figs, reports, scores, train_len, test_len):

    files = os.listdir(r'logging')
    name = type(model).__name__
    if 'experiments_summary.csv' not in files:
        experiment = 1
        df = pd.DataFrame({'experiment': [experiment],
                           'model': [name],
                           'batch_size': [bs],
                           'optimizer': [optimizer],
                           'learning_rate': [lr],
                           'epochs': [epochs],
                           'acc_train': [scores[0]],
                           'acc_test': [scores[1]]})
    else:
        df = pd.read_csv(r'logging/experiments_summary.csv')
        experiment = df['experiment'].max() + 1
    torch.save(model.state_dict(), f'logging/XP{experiment}_model')
    df.loc[len(df)] = [experiment,
                       name,
                       bs,
                       optimizer,
                       lr,
                       epochs,
                       round(scores[0], 2),
                       round(scores[1], 2),
                       train_len,
                       test_len]
    figs[0].savefig(f'logging/XP{experiment}_loss.jpg')
    figs[1].savefig(f'logging/XP{experiment}_accuracy.jpg')
    train, test = reports
    df.to_csv('logging/experiments_summary.csv', index=False)
    for report, split in zip((train, test), ('train', 'test')):
        with open(fr'logging/XP{experiment}_{split}.csv', 'w') as f:
            f.write(classification_report_df(report))
