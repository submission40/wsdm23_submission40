import numpy as np
import argparse
from src.data_manager.data_manager import DataManager
import seaborn as sns
import matplotlib.pyplot as plt
import os
def confidence_interval(metrics):
  n = metrics.shape[0]
  std = metrics.std()
  return 1.96 * (std/np.sqrt(n))

def create_grouping_matrix():
    # multiplying by this matrix gives a grouped average
    M = np.zeros((10000, 10))
    kernel = np.ones((1,1000))/ 1000
    for i in range(10):
      M[1000*i: 1000* (i+1), i] = kernel
    return M

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type = str, required = True,
                    help = "metric to compute")
    parser.add_argument("--recos_path", type = str, required = False,
                    help = "path to recos", default="ressources/recos")
    parser.add_argument("--plots_path", type = str, required = False,
                    help = "path to plots", default="ressources/plots")
    parser.add_argument("--models", type = str, required = False,
                    help = "comma separated names of models to evaluate", default="sknn,vsknn,stan,vstan,AVG,CNN,GRU,Dec,Dec-FM,Dec-NN")

    args = parser.parse_args()
    model_names = args.models.split(",")
    l = len(model_names)
    os.makedirs(args.plots_path, exist_ok=True)
    recos = [np.load(("%s/%s.npy") % (args.recos_path, m)) for m in model_names]
    sns.set()
    sns.set_palette("bright")
    cp = sns.color_palette()
    data_manager = DataManager()
    test_evaluator, test_dataloader = data_manager.get_test_data("test")
    M = create_grouping_matrix()
    for i in range(l):
        m = model_names[i]
        rec = recos[i]
        if args.metric== "coverage": # coverage can not be averaged so no confidence interval
            metrics = test_evaluator.compute_cov(rec)
            print("%s: %.4f" % (m, np.mean(metrics)))
        else:
            if args.metric == "recall":
                metrics = test_evaluator.compute_all_recalls(rec)
            if args.metric == "ndcg":
                metrics = test_evaluator.compute_all_ndcgs(rec)
            if args.metric == "clicks":
                metrics = test_evaluator.compute_all_clicks(rec)
            if args.metric == "precision":
                metrics = test_evaluator.compute_all_precisions(rec)
            if args.metric == "r-precision":
                metrics = test_evaluator.compute_all_R_precisions(rec)
            if args.metric== "popularity":
                metrics = test_evaluator.compute_norm_pop(rec)
            alpha = confidence_interval(metrics)
            lower = np.mean(metrics) - alpha
            upper = np.mean(metrics) + alpha
            groups = metrics.dot(M)
            sns.lineplot(x=data_manager.N_SEED_SONGS, y=groups, label=m, markers=True, linewidth=1.1, color=cp[i])
            print("%s: %.4f-%.4f" % (m, np.mean(metrics), alpha))
            print((metrics == 0).sum())
            plt.xlabel("number of seed tracks", {"size": 14, 'weight': 'bold'})
            plt.ylabel(args.metric, {"size": 14, 'weight': 'bold'})
            plt.xticks(fontsize=12, ticks=data_manager.N_SEED_SONGS)
            plt.yticks(fontsize=12)
            plt.legend(loc="best")
            plt.savefig("%s/%s.pdf" % (args.plots_path, args.metric), bbox_inches = "tight")
            plt.show()