import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class CloseToDots:
    def __init__(self,featuresPoses,featuresNames) -> None:
        """
        param: featuresPoses array of n dots
        param: featuresNames array of n string
        """
        assert len(featuresPoses)==len(featuresNames),"featuresPoses,featuresNames must same lenght"
        
        self.df = pd.DataFrame(featuresPoses)

        self.featuresNames=featuresNames



    def show_dots(self,features_powers:np.ndarray):
        if(features_powers.min()<0):
            features_powers-= (features_powers.min()-0)
        features_powers=(features_powers+1)**4
        features_powers= features_powers/features_powers.sum()
        newDot = ((self.df[0]*features_powers).sum(),(self.df[1]*features_powers).sum())
        
        
        plt.clf()
        # show names
        for i,(x,y) in enumerate(self.df.values):
            plt.text(x,y,self.featuresNames[i],)
        plt.scatter(self.df[0],self.df[1])
        plt.scatter(newDot[0],newDot[1],marker="x")

        plt.pause(0.0001)
    def show(self):plt.show()


