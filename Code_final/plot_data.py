from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_positions(x,T):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(x[:,0], x[:,1], x[:,2], color = "green")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.suptitle("Particles position for "+str(T)+" time")
    #plt.title("Particles position for %i time", %T)
    plt.savefig("Particles_position_for_"+str(T)+"_time30.png")
    plt.close()

