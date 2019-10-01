import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("training_finetune.csv")

sns.set_palette(sns.color_palette("GnBu_d"))
fig = plt.figure()
ax1 = fig.add_subplot(111)
sns.lineplot(x='epochs', y=" train_acc", ci=None, ax = ax1, data=data)
sns.lineplot(x='epochs', y=" val_acc", ci=None, ax = ax1, data=data)

sns.set_palette(sns.color_palette("BuGn_r"))
ax2 = ax1.twinx()
sns.lineplot(x='epochs', y=" train_loss", ax=ax2, data=data)
sns.lineplot('epochs', " val_loss", ax=ax2, data = data)

fig.legend(["Train Acc","Val Acc", "Train Loss", "Val Loss"], loc=1,
             bbox_to_anchor=(1,0.75), bbox_transform=ax1.transAxes)

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax2.set_ylabel("Loss")
plt.show()