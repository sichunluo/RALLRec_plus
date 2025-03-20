import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from matplotlib.ticker import PercentFormatter

def load_data(json_files, txt_files, ground_truth_path):
    with open(json_files[0], 'r') as f:
        num_samples = len(json.load(f)['samples'])
    
    num_experiments = len(json_files)
    lengths = np.zeros((num_samples, num_experiments), dtype=int)
    preds = np.zeros((num_experiments, num_samples))
    
    for exp_idx, json_path in enumerate(json_files):
        with open(json_path, 'r') as f:
            data = json.load(f)
            for sample_idx, sample in enumerate(data['samples']):
                lengths[sample_idx, exp_idx] = len(sample.get('pred', ''))
    
    for exp_idx, txt_path in enumerate(txt_files):
        with open(txt_path, 'r') as f:
            preds[exp_idx] = [float(line.strip()) for line in f]
    
    golds=[]
    with open(ground_truth_path, 'r') as f:
        data = json.load(f)
        for idx, sample in enumerate(data):
            golds.append(sample['output'])
    golds=golds[:num_samples]
    golds = [int(data_point == 'Yes.') for data_point in golds]

    return lengths, preds, golds

def compute_metrics(preds, golds):
    return {
        'auc': roc_auc_score(golds, preds),
        'logloss': log_loss(golds, preds),
        'acc': accuracy_score(golds, (preds > 0.5).astype(int))
    }

def compute_experiment_metrics(preds, golds):
    experiment_metrics = []
    
    assert preds.shape[1] == len(golds)
    
    for exp_idx in range(preds.shape[0]):
        exp_pred = preds[exp_idx]
        metrics = compute_metrics(exp_pred, golds)
        metrics['experiment'] = exp_idx + 1
        experiment_metrics.append(metrics)
    
    return experiment_metrics

def plot_consistency(preds, golds, window_size=500):
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    variances = np.var(preds, axis=0)
    sorted_indices = np.argsort(variances)
    sorted_correct = (np.mean(preds, axis=0) > 0.5).astype(int) == np.array(golds)
    sorted_correct = sorted_correct[sorted_indices].astype(int)
    
    cum_correct = np.cumsum(sorted_correct)
    moving_acc = (cum_correct[window_size:] - cum_correct[:-window_size]) / window_size
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    line1 = ax1.plot(moving_acc, color='#1f77b4', lw=2, label='Accuracy')
    ax1.set_ylabel('Accuracy', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(np.sort(variances)[window_size//2:-window_size//2], 
                    color='#ff7f0e', alpha=0.7, lw=1.5, label='Variance')
    ax2.set_ylabel('Variance', color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, np.max(variances)*1.1)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=12)
    
    plt.savefig('consensus_analysis.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    dataset={0:"ml-1m-r1",1:"amazon-movies-r1",2:"BookCrossing-r1"}
    dataset=dataset[0]
    print(dataset)

    if dataset == "amazon-movies-r1":
        json_files = [
        "path/to/file",
        ]
        txt_files = [f.replace('.json', '.txt') for f in json_files]
        ground_truth_path = "path/to/file"

        txt_files_llm = [
        "path/to/file",
        ]

    elif dataset == "BookCrossing-r1":
        json_files = [
        "path/to/file",
        ]
        txt_files = [f.replace('.json', '_dsllama8b.txt') for f in json_files]
        ground_truth_path = "path/to/file"
        
        txt_files_llm = [
        "path/to/file",
        ]

    elif dataset == "ml-1m-r1":
        json_files = [
        "path/to/file",
        ]
        txt_files = [f.replace('.json', '_dsllama8b.txt') for f in json_files]
        ground_truth_path = "path/to/file"

        txt_files_llm = [
        "path/to/file",
        ]

    if 'Book' in dataset:
        num_samples=873
    else:
        num_samples=10000

    num_experiments = len(txt_files_llm)
    preds_llm = np.zeros((num_experiments, num_samples))
    for exp_idx, txt_path in enumerate(txt_files_llm):
        with open(txt_path, 'r') as f:
            preds_llm[exp_idx] = [float(line.strip()) for line in f]

    lengths, preds, golds = load_data(json_files, txt_files, ground_truth_path)

    def fused_prediction(model_a_preds, model_b_preds, w_A=0.1,epsilon = 1e-3, w_B=1., tau_A=1.0, tau_B=1.):
        mu_a = np.mean(model_a_preds)
        var_a = np.var(model_a_preds)

        mu_b = np.mean(model_b_preds)
        mu_b = model_b_preds[0]
        var_b = np.var(model_b_preds)
        
        weight_a = w_A / (var_a / tau_A + epsilon)
        weight_b = w_B / (var_b / tau_B + epsilon)
        
        fused = (weight_a * mu_a + weight_b * mu_b) / (weight_a + weight_b)
        return fused

    res=[]
    for i in range(num_samples):
        res.append(fused_prediction(preds[:,i], preds_llm[:,i], w_A=0.1, epsilon=1e-3))
    print( compute_metrics(np.array(res), golds) )

    for i_ in [1e-2,1e-3,1e-4,1e-5,1e-6]:
        print(i_)
        res=[]
        for i in range(num_samples):
            res.append(fused_prediction(preds[:,i], preds_llm[:,i], w_A=0.1, epsilon=i_))
        print( compute_metrics(np.array(res), golds) )
