from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

ITERATIONS_TYPES = ['macro', 'micro', 'inner']

def plot_convergence_statistics(results_dict: Dict[str, List[Tuple[float, float]]]) -> plt.figure:
  fig = plt.figure(constrained_layout=True, figsize=(10, 4))
  subplots = fig.subplots(1, 3)

  matplotlib.rc('xtick', labelsize=20)  
  matplotlib.rc('ytick', labelsize=20)

  for idx, iteration_type in enumerate(ITERATIONS_TYPES):
    iterations = [results_dict[key][idx][0] for key in results_dict.keys()]
    deviations = [results_dict[key][idx][1] for key in results_dict.keys()]
    subplots[idx].errorbar(np.arange(len(iterations)), iterations, yerr=deviations, fmt='o', capsize=5)
    subplots[idx].set_ylabel('N iterations')
    subplots[idx].set_xlabel('Method')
    subplots[idx].set_title(iteration_type)
    subplots[idx].set_xticks(np.arange(len(iterations)), list(results_dict.keys()))

  return fig

if __name__ == "__main__":
  results_dict = {
    'ao_min': [(5.7, 1.453), (17.8, 5.689), (71.45, 29.135)],
    'ML-MO': [(10.0, 5), (34.65, 17), (163.9, 80)],
    'ML-F': [(3.55, 0.497), (10.1, 1.578), (32.3, 7.950)]
  }

  fig = plot_convergence_statistics(results_dict)
  plt.show()