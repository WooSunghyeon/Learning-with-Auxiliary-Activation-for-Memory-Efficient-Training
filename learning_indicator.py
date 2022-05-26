import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
parser.add_argument('--model_path', type=str, help='The path to the saved model file to evaluation')

def main():
    
    data = np.load(args.model_path).flatten()
    print(data.nbytes)   
    daa_pos = data[data>0]
    daa_neg = data[data<0]
    
    fig1 = plt.figure(1)
    cmap = plt.cm.get_cmap('hot_r')
    cmap2 = plt.cm.get_cmap('ocean')
    axes1 = fig1.add_axes([0.2,0.2,0.7,0.7])
    
    xmax=1
    plt.xticks(np.array([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]),labels=['-1', '', '-0.5', '','0','','0.5','', '1'], fontsize=15)
    #if args.learning_rule == 'sa1' or args.learning_rule == 'sa2' or args.learning_rule == 'sa3' or args.learning_rule == 'sa4':
        #xmax=0.1
        #plt.xticks(np.array([-0.1,-0.075,-0.05,-0.025,0,0.025,0.05,0.075,0.1]),labels=['-0.1', '', '-0.05', '','0','','0.05','', '0.1'], fontsize=15)
    plt.yticks(fontsize=15)
    
    n, bins, patches = axes1.hist(daa_pos, bins = 1000, color = 'green', range=(-xmax,xmax))
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    maxi = np.abs(bin_centers).max()
    norm = plt.Normalize(-maxi,maxi)
    for c, p in zip(bin_centers, patches):
            plt.setp(p, "facecolor", cmap(norm(c)))
        
    n2, bins2, patches2 = axes1.hist(daa_neg, bins = 1000, color = 'green', range=(-xmax,xmax))
    bin_centers2 = 0.5*(bins2[:-1]+bins2[1:])
    maxi2 = np.abs(bin_centers2).max()
    norm2 = plt.Normalize(-maxi2,maxi2)
    for c, p in zip(bin_centers2, patches2):
            plt.setp(p, "facecolor", cmap2(norm2(c)))
        
    #cbar1 = fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap), location = 'bottom', shrink = 0.3)
    #cbar1.mappable.set_clim(-1,1)
    plt.xlim([-xmax,xmax])
    plt.ylim([0.7,10**5])
    plt.xlabel("Learning Indicator", weight='bold', fontsize = 18)
    plt.ylabel("Distribution", weight='bold', fontsize = 18)
    plt.yscale('log')
    plt.show()
    
if __name__ == '__main__':
    main()      