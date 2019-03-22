import talos as ta
import matplotlib

r = ta.Reporting('mlp_mnist_hyperparams.csv')

print(r.data)

r.plot_line(metric="categorical_accuracy")
