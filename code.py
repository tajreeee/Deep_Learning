import pandas as pd
import numpy as np
import math


heart_data = pd.read_csv("heart.csv")


features = heart_data.drop("target", axis=1).values
labels = heart_data["target"].values


params = np.array([0.02] * features.shape[1])
offset = 0.1


alpha = 0.0000005


def activation(value):
    return 1 / (1 + math.exp(-value))


for cycle in range(5):

    print(f"\nTraining Cycle {cycle + 1}")

    for row in range(len(features)):

        sample = features[row]
        actual = labels[row]

       
        net_input = np.dot(params, sample) + offset
        output = activation(net_input)

        loss = output - actual

       
        gradient = 2 * loss * output * (1 - output)

       
        params = params - alpha * gradient * sample
        offset = offset - alpha * gradient

        print(
            f"Output={output:.3f}, "
            f"Loss={loss:.3f}"
        )

print("\nLearned Parameters")

for index, value in enumerate(params):
    print(f"weight_{index + 1} =", round(value, 4))

print("bias =", round(offset, 4))