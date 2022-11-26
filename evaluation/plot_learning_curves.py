import numpy as np
import matplotlib.pyplot as plt

macro_iterations = {
    'ao-min': (7.69, 1.22),
    'hartree-fock': (7.04, 1.25),
    's005': [(7.08, 1.83), (5.4, 1.33), (5.28, 1.48)],
    's01': [(5.24, 1.36), (5.24, 1.36), (5.04, 1.28)]
}
micro_iterations = {
    'ao-min': (26.2, 4.66),
    'hartree-fock': (23.52, 4.60),
    's005': [(23.52, 4.60), (17.36, 5.25), (16.72, 5.52)],
    's01': [(16.68, 7.27), (16.68, 7.27), (15.64, 4.95)]
}
inner_iterations = {
    'ao-min': (112.48, 23.37),
    'hartree-fock': (100.88, 24.53),
    's005': [(101.72, 35.68), (68.52, 28.23), (65.4, 27.68)],
    's01': [(65.12, 26.29), (65.12, 26.29), (59.04, 24.9)]
}
casci_error = {
    'ao-min': (0.14, 0.02),
    'hartree-fock': (0.085, 0.026),
    's005': [(0.14, 0.08), (0.056, 0.032), (0.048, 0.028)],
    's01': [(0.05, 0.025), (0.05, 0.025), (0.035, 0.024)]
}



if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, figsize=(8,8))

    # macro iterations
    for method in ['s005', 's01']:
        axs[0, 0].errorbar(x=[200, 1000, 5000],
                           y=[val[0] for val in macro_iterations[method]],
                           fmt='x',
                           capsize=4)
    axs[0, 0].plot(np.arange(6000), 
                    [macro_iterations['ao-min'][0]] * 6000,
                    ls='--',
                    color='purple')
    axs[0, 0].plot(np.arange(6000), 
                    [macro_iterations['hartree-fock'][0]] * 6000,
                    ls='--',
                    color='red')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlim([100, 6000])
    axs[0, 0].set_title('Macro iterations')


    # micro_iterations
    for method in ['s005', 's01']:
        axs[0, 1].errorbar(x=[200, 1000, 5000],
                           y=[val[0] for val in micro_iterations[method]],
                           fmt='x',
                           capsize=4)
    axs[0, 1].plot(np.arange(6000), 
                    [micro_iterations['ao-min'][0]] * 6000,
                    ls='--',
                    color='purple')
    axs[0, 1].plot(np.arange(6000), 
                    [micro_iterations['hartree-fock'][0]] * 6000,
                    ls='--',
                    color='red')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlim([100, 6000])
    axs[0, 1].set_title('Micro iterations')


    # inner_iterations
    for method in ['s005', 's01']:
        axs[1, 0].errorbar(x=[200, 1000, 5000],
                           y=[val[0] for val in inner_iterations[method]],
                           fmt='x',
                           capsize=4)
    axs[1, 0].plot(np.arange(6000), 
                    [inner_iterations['ao-min'][0]] * 6000,
                    ls='--',
                    color='purple')
    axs[1, 0].plot(np.arange(6000), 
                    [inner_iterations['hartree-fock'][0]] * 6000,
                    ls='--',
                    color='red')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlim([100, 6000])
    axs[1, 0].set_title('Inner iterations')


    # casci_error
    line1 = axs[1, 1].errorbar(x=[200, 1000, 5000],
                           y=[val[0] for val in casci_error['s005']],
                           fmt='x',
                           capsize=4)
    line2 = axs[1, 1].errorbar(x=[200, 1000, 5000],
                           y=[val[0] for val in casci_error['s01']],
                           fmt='x',
                           capsize=4)
    line3 = axs[1, 1].plot(np.arange(6000), 
                    [casci_error['ao-min'][0]] * 6000,
                    ls='--',
                    color='purple')
    line4 = axs[1, 1].plot(np.arange(6000), 
                    [casci_error['hartree-fock'][0]] * 6000,
                    ls='--',
                    color='red')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlim([100, 6000])
    axs[1, 1].set_title('Casci error (Ha)')

    plt.legend(
        handles=[line3[0], line4[0], line1, line2],
        labels=['ao-min', 'hartree-fock', 's005-phisnet', 's01-phisnet'],
        loc='upper right', bbox_to_anchor=(1.2, 1)
    )
    plt.savefig('test.png')