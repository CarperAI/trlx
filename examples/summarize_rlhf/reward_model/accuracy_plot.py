import pickle

from matplotlib import pyplot as plt

result = pickle.load(open("reward_res.pkl", "rb"))

# calculate the average reward for each 10000 samples

batch_size = 10000
lst_acc = []
lst_index = []
for i in range(0, len(result), batch_size):

    if i + batch_size > len(result):
        sub_result = result[i:]
        lst_index.append(len(result))
    else:
        lst_index.append(i + batch_size)
        sub_result = result[0 : i + batch_size]
    avg_reward = sum(sub_result) / len(sub_result)
    lst_acc.append(avg_reward)

import matplotlib.pyplot as plt

# Assume lst_index and lst_acc are defined and contain the data you want to plot

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data
ax.plot(lst_index, lst_acc, "o-b", label="Accuracy on Validation")

# Add a legend and title
ax.legend()
ax.set_title("Accuracy vs. Number of Examples")

# Add x and y labels
ax.set_xlabel("Number of Examples")
ax.set_ylabel("Accuracy")

# Save the figure to a file
fig.savefig("accuracy_plot.png")
