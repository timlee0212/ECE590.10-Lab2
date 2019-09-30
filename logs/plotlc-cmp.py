import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

bn01 = pd.read_csv("training_lr0.1_bn.csv")
bn005 = pd.read_csv("training_lr0.05_bn.csv")
bn001 = pd.read_csv("training_lr0.01_bn.csv")

sns.set_style("whitegrid")

sns.set_palette(sns.color_palette("Paired"))
fig = plt.figure()
sns.lineplot(x='epochs', y=" train_acc", ci=None, data=bn01)
sns.lineplot(x='epochs', y=" val_acc", ci=None, data=bn01)
sns.lineplot(x='epochs', y=" train_acc", ci=None, data=bn005)
sns.lineplot(x='epochs', y=" val_acc", ci=None, data=bn005)
sns.lineplot(x='epochs', y=" train_acc", ci=None, data=bn001)
sns.lineplot(x='epochs', y=" val_acc", ci=None, data=bn001)
plt.legend(["Train 0.1", "Val 0.1", "Train 0.05", "Val 0.05", "Train 0.01", "Val 0.01"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()

fig = plt.figure()
sns.lineplot(x='epochs', y=" train_loss", ci=None, data=bn01)
sns.lineplot(x='epochs', y=" val_loss", ci=None, data=bn01)
sns.lineplot(x='epochs', y=" train_loss", ci=None, data=bn005)
sns.lineplot(x='epochs', y=" val_loss", ci=None, data=bn005)
sns.lineplot(x='epochs', y=" train_loss", ci=None, data=bn001)
sns.lineplot(x='epochs', y=" val_loss", ci=None, data=bn001)
plt.legend(["Train 0.1", "Val 0.1", "Train 0.05", "Val 0.05", "Train 0.01", "Val 0.01"])
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()