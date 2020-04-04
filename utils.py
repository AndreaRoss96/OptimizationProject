
import matplotlib.pyplot as plt
import seaborn as sns

class Printer:
    def __init__(self):
        figure = plt.figure(figsize=(10,10))
        pass

    def plotDif(self, pred_o, true_o, feat_1, feat_2):
        sns.scatterplot(feat_1, feat_2, style =list(pred_o==true_o), hue= list(true_o));
