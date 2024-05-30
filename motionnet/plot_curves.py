"""
author : VIctoria
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def main():

    #path_csv  = "/home/sdi-2023-01/metrics.csv"
    #path_csv = "lightning_logs/version_30/metrics.csv"
    path_csv = "/home/sdi-2023-01/Téléchargements/metrics.csv"
    #recuperere "version_valentinV1" par l'avant derniere occurence de "/"
    name = path_csv.split("/")[-2]

    df = pd.read_csv(path_csv)

    list_values = df.columns
    list_values = list_values[2:]

    print(list_values)

    #plot minADE6 pour train et val

    value_metric = "minADE6"

    fig, axs = plt.subplots(1, 1, figsize=(50, 25))

    fig.suptitle('Metrics for train and val') #4 premiere metrics pour train et val

    metric_train = "train/"+value_metric

    metric_val = "val/"+value_metric

    metric_train_values = df[metric_train]
    metric_val_values = df[metric_val]

    #enlever les valeurs nan et la ligne correspondante
    metric_train_values = metric_train_values.dropna()
    metric_val_values = metric_val_values.dropna()

    #remplacer les valeurs 1er colonne par des entiers de 0 a n
    metric_train_values.index = range(len(metric_train_values))
    metric_val_values.index = range(len(metric_val_values))

    axs.plot(metric_train_values, label = metric_train)
    axs.plot(metric_val_values, label = metric_val)

    axs.legend(prop={'size': 30})
    axs.set_title(metric_train, fontsize=30)

    #xlabel et ylabel en taille 30
    axs.set_xlabel("epoch", fontsize=30)
    axs.set_ylabel(metric_train, fontsize=30)

    #taille des axes en 25
    axs.tick_params(axis='both', which='major', labelsize=25)

    #show
    plt.show()


    ### print last value of minADE6 for train and val
  
    print("minADE6 for train")
    print(df["train/minADE6"].iloc[-1])

    

    
    ##### train value = half of the values
    list_values_for_train = list_values[:len(list_values)//2] 
    list_values_for_val = list_values[len(list_values)//2:]


    #################### Pour chaque metrics (il y en a 8 pour val et 8 pour train), plot la courbe pour val et train sur un meme graphe : il y aura donc 8 figures
    #donc faire subplot(4,2,i) pour i allant de 1 a 8

    fig, axs = plt.subplots(4, 2, figsize=(50, 25))

 

    fig.suptitle('Metrics for train and val') #4 premiere metrics pour train et val

    for i in range(0, len(list_values_for_train)//2):
        metric_train = list_values_for_train[i]
        metric_val = list_values_for_val[i]


        metric_train_values = df[metric_train]
        metric_val_values = df[metric_val]

        #enlever les valeurs nan et la ligne correspondante
        metric_train_values = metric_train_values.dropna()
        metric_val_values = metric_val_values.dropna()

        #remplacer les valeurs 1er colonne par des entiers de 0 a n
        metric_train_values.index = range(len(metric_train_values))
        metric_val_values.index = range(len(metric_val_values))

        

        axs[i//2, i%2].plot(metric_train_values, label = metric_train)
        axs[i//2, i%2].plot(metric_val_values, label = metric_val)
        axs[i//2, i%2].legend(prop={'size': 30})
        axs[i//2, i%2].set_title(metric_train, fontsize=30)

        #xlabel et ylabel en taille 30
        axs[i//2, i%2].set_xlabel("epoch", fontsize=30)
        axs[i//2, i%2].set_ylabel(metric_train, fontsize=30)

        #taille des axes en 25
        axs[i//2, i%2].tick_params(axis='both', which='major', labelsize=25)
        

    #save
    plt.savefig(name+"_metrics_part1.png")
    

    # 4 metriques suivantes
    fig, axs = plt.subplots(4, 2, figsize=(50, 25))

    fig.suptitle('Metrics for train and val') #4 premiere metrics pour train et val

    for i in range(len(list_values_for_train)//2 , len(list_values_for_train)):
        metric_train = list_values_for_train[i]
        metric_val = list_values_for_val[i]


        metric_train_values = df[metric_train]
        metric_val_values = df[metric_val]

        #enlever les valeurs nan et la ligne correspondante
        metric_train_values = metric_train_values.dropna()
        metric_val_values = metric_val_values.dropna()

        #remplacer les valeurs 1er colonne par des entiers de 0 a n
        metric_train_values.index = range(len(metric_train_values))
        metric_val_values.index = range(len(metric_val_values))

        #i va de 4 a 8 !
        axs[(i-8)//2, (i-8)%2].plot(metric_train_values, label = metric_train)
        axs[(i-8)//2, (i-8)%2].plot(metric_val_values, label = metric_val)
        axs[(i-8)//2, (i-8)%2].legend(prop={'size': 30})
        axs[(i-8)//2, (i-8)%2].set_title(metric_train, fontsize=30)

        #xlabel et ylabel en taille 30
        axs[(i-8)//2, (i-8)%2].set_xlabel("epoch", fontsize=30)
        axs[(i-8)//2, (i-8)%2].set_ylabel(metric_train, fontsize=30)

        axs[(i-8)//2, (i-8)%2].tick_params(axis='both', which='major', labelsize=25)

    #save
    plt.savefig(name+"_metrics_part2.png")


    ###### print minADE6 derniere valeur
    print("minADE6 for val")
    print(df["val/minADE6"].iloc[-1])

    print("minADE6 for train")
    print(df["train/minADE6"].iloc[-1])
    

    
if __name__=="__main__":
    main()
