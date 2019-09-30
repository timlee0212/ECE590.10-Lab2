import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

a1 = pd.read_csv("training_lr0.1.csv")
a2 = pd.read_csv("training_lr0.1_bn.csv")

sns.set_style("whitegrid")

sns.set_palette(sns.color_palette("coolwarm", 4))
fig = plt.figure()
sns.lineplot(x='epochs', y=" train_acc", ci=None, data=a1)
sns.lineplot(x='epochs', y=" val_acc", ci=None, data=a1)
sns.lineplot(x='epochs', y=" train_acc", ci=None, data=a2)
sns.lineplot(x='epochs', y=" val_acc", ci=None, data=a2)
plt.legend(["Train Acc","Val Acc", "Train Acc w\BN", "Val Acc w\BN"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()

fig = plt.figure()
sns.lineplot(x='epochs', y=" train_loss", ci=None, data=a1)
sns.lineplot(x='epochs', y=" val_loss", ci=None, data=a1)
sns.lineplot(x='epochs', y=" train_loss", ci=None, data=a2)
sns.lineplot(x='epochs', y=" val_loss", ci=None, data=a2)
plt.legend(["Train Loss","Val Loss", "Train Loss w\BN", "Val Loss w\BN"])
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()