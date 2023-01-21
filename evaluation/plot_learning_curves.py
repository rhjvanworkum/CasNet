import numpy as np
import matplotlib.pyplot as plt

# """ STO-6G """
# macro_iterations = {
#     'ao-min': (7.69, 1.22),
#     'hartree-fock': (7.04, 1.25),
#     's005': [(7.08, 1.83), (5.4, 1.33), (5.28, 1.48)],
#     's01': [(5.24, 1.36), (4.88, 1.37), (5.04, 1.28)]
# }
# micro_iterations = {
#     'ao-min': (26.2, 4.66),
#     'hartree-fock': (23.52, 4.60),
#     's005': [(23.52, 4.60), (17.36, 5.25), (16.72, 5.52)],
#     's01': [(16.68, 7.27), (15.36, 5.39), (15.64, 4.95)]
# }
# inner_iterations = {
#     'ao-min': (112.48, 23.37),
#     'hartree-fock': (100.88, 24.53),
#     's005': [(101.72, 35.68), (68.52, 28.23), (65.4, 27.68)],
#     's01': [(65.12, 26.29), (59.24, 26.77), (59.04, 24.9)]
# }
# casci_error = {
#     'ao-min': (0.14, 0.02),
#     'hartree-fock': (0.085, 0.026),
#     's005': [(0.14, 0.08), (0.056, 0.032), (0.048, 0.028)],
#     's01': [(0.05, 0.025), (0.038, 0.025), (0.035, 0.024)]
# }

""" cc-pVDZ """
macro_iterations = {
    'ao-min': (9.32, 2.66),
    'hartree-fock': (8.92, 3.32),
    's005': [(34.52, 6.95), (35.48, 6.95), (35.2, 8.73)],
    's01': [(33.84, 8.15), (32.64, 7.85), (16.6, 11.22)]
}
micro_iterations = {
    'ao-min': (33.2, 10.38),
    'hartree-fock': (31.56, 13.24),
    's005': [(130.04, 29.57), (129.92, 21.27), (130.04, 31.93)],
    's01': [(125.12, 29.17), (119.04, 26.84), (61.88, 44.02)]
}
inner_iterations = {
    'ao-min': (147.44, 48.94),
    'hartree-fock': (141.16, 65.28),
    's005': [(678.52, 154.45), (668.68, 100.16), (661.64, 162.17)],
    's01': [(639.72, 152.75), (603.12, 136.67), (308.36, 231.93)]
}
casci_error = {
    'ao-min': (0.15, 0.2),
    'hartree-fock': (0.09, 0.013),
    's005': [(10.68, 3.64), (12.37, 3.28), (7.79, 2.86)],
    's01': [(8.33, 3.08), (6.76, 2.73), (1.76, 1.96)]
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
    # axs[0, 0].set_yscale('log')
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
    # axs[0, 1].set_yscale('log')
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
    # axs[1, 0].set_yscale('log')
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
    # axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlim([100, 6000])
    axs[1, 1].set_title('Casci error (Ha)')

    plt.legend(
        handles=[line3[0], line4[0], line1, line2],
        labels=['ao-min', 'hartree-fock', 's005-phisnet', 's01-phisnet'],
        loc='upper right', bbox_to_anchor=(1.2, 1)
    )
    plt.show()