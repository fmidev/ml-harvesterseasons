import matplotlib.pyplot as plt
import pandas as pd


#data1 = pd.read_csv("/home/ubuntu/data/ML/results/OCEANIDS/models-validation-rmse.csv")

#plot = data1.plot(kind="bar", title="RMSE values")
#plot.set_xticklabels(["Vuosaari", "Raahe", "Rauma","Malaga"], rotation=0)
#plot.legend(loc="upper left", bbox_to_anchor=(1,1))
#plot.get_figure().savefig("plot_rmse_values.png", bbox_inches="tight")


data = pd.read_csv("/home/ubuntu/data/ML/results/OCEANIDS/models-validation-mae.csv")

plot = data.plot(kind="bar", title="MAE values")
plot.set_xticklabels(["Vuosaari", "Raahe", "Rauma", "Malaga"], rotation=0)
plot.legend(loc="upper left", bbox_to_anchor=(1,1))
plot.get_figure().savefig("plot_mae_values.png", bbox_inches="tight")