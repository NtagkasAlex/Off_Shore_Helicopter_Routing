import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__ == "__main__":

    plt.rcParams['figure.figsize'] = [15.3, 12]
    # plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.style.use('bmh')

    matplotlib.rcParams.update({'grid.linewidth': 1.5, "grid.color": "#000000", "grid.alpha": "0.2"})
    matplotlib.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    # prefix = "runs/imitation_random_start/"; name = "imitation_fixed.pdf" ; title = "Imitation Learning: Achieve orientation with fixed initial orientation"
    title = "Timing Round Off"
    # platforms=[5,10,20,30,40,51]
    
    platforms=[5,10,15,20,30,40]#,5,5,5,5]
    names=["N="+str(N) for N in platforms]

    paths=["times-"+str(n)+".npy" for n in platforms]
    # paths=["times_cg-"+str(n)+".npy" for n in platforms]

    times=[]
    for i, p in enumerate(paths):
        losses = np.load(p)
        print(losses)
        print(p)
        times.append(losses)

    times=np.array(times).T
    x_ticks = np.array(platforms)

    ax.plot(x_ticks, np.median(times, axis=0), linewidth=2.5, color=colors[i])
    # plt.fill_between(np.arange(len(losses[0])), np.median(losses, axis=0)-np.std(losses, axis=0).astype(np.float32), np.median(losses, axis=0) + np.std(losses, axis=0).astype(np.float32), alpha=0.3, label='_nolegend_')
    ax.fill_between(x_ticks, np.percentile(times, 25, axis=0), np.percentile(times, 75, axis=0), alpha=.2, label='_nolegend_', color=colors[i])


    # plt.plot(r)
    ax.set_ylabel('Time in seconds', fontdict=dict(weight='bold', size="22"))
    ax.set_xlabel('Number of Platforms', fontdict=dict(weight='bold', size="22"))

    ax.set_title(title, fontdict=dict(weight='bold', size=25))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # ax.legend(names, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, prop=dict(weight='normal', size=25))
    plt.tight_layout()
    # plt.savefig(".tmp/"+name)
    # plt.ticklabel_format(useOffset=False)

    plt.show()

