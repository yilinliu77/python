import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    # plt.figure()
    # plt.subplot(2,1,1)
    # x = np.random.randint(0,500,100)
    # y = np.random.standard_normal(100)
    # plt.scatter(x,y,color="g")
    # plt.xlabel("Random integers")
    # plt.ylabel("Random normal dist.")
    # plt.title("A beautiful Scatter Plot of the Standard normal distribution")
    #
    # plt.subplot(2, 1, 2)
    # x = np.arange(0,11,0.01)
    # y = np.sin(x)
    # plt.plot(x, y)
    # plt.xlabel("x")
    # plt.ylabel("sin(x)")
    # plt.title("Visualizing Sin")
    #
    # plt.tight_layout()
    # plt.show()

    dataset=pd.read_csv("titanicTrain.csv")
    plt.figure()
    plt.subplot(1, 2, 1)
    class1 = dataset[dataset["Pclass"] == 1]["Age"].to_numpy()
    class2 = dataset[dataset["Pclass"] == 2]["Age"].to_numpy()
    class3 = dataset[dataset["Pclass"] == 3]["Age"].to_numpy()
    class1=class1[~np.isnan(class1)]
    class2=class2[~np.isnan(class2)]
    class3=class3[~np.isnan(class3)]

    plt.hist(class1,bins=[0,10,20,30,40,50,60,70,80],color=(0,0,1),label="1st",alpha=0.1)
    plt.hist(class2,bins=[0,10,20,30,40,50,60,70,80],color=(1,0,0),label="2nd",alpha=0.1)
    plt.hist(class3,bins=[0,10,20,30,40,50,60,70,80],color=(0,1,0),label="3rd",alpha=0.1)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Passenger Class and Age Distribution")

    plt.subplot(1, 2, 2)
    sizes=[(dataset["Pclass"] == 1).sum(),(dataset["Pclass"] == 2).sum(),(dataset["Pclass"] == 3).sum()]
    plt.pie(sizes,autopct='%1.1f%%',labels=["1","2","3"],explode=[0.1,0.1,0.1])
    plt.title("Class Percentage")

    plt.tight_layout()
    plt.show()


