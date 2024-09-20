import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # platforms=[10,20,30,40,51]

    platforms=[5,10,20,30,40,51]
    names=["N="+str(N) for N in platforms]
    plots=[]
    for n in platforms:
        res=np.load("results-"+str(n)+".npy")
        plots.append(res)
    # print(res)
    plt.boxplot(plots, showfliers=False)
    plt.xticks([i+1 for i in range(len(platforms))] , names)  # Label for each box



    plt.show()




