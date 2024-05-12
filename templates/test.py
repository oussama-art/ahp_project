weighted_matrix = [[[0.01692814, 0.02821356, 0.02257085]],
                   [[0.01374401, 0.00859001, 0.01546201]],
                   [[0.00313689, 0.00224064, 0.00268876]],
                   [[0.07743496, 0.11615243, 0.15486991]],
                   [[0.04881956, 0.0781113, 0.06834739]],
                   [[0.0088114, 0.01586053, 0.01057369]],
                   [[0.21257553, 0.10628776, 0.26571941]],
                   [[0.07658864, 0.15317728, 0.0957358]],
                   [[0.03085714, 0.01028571, 0.01542857]]]

# List to store all weights for each alternative
all_weights = []

for alternative_weights in weighted_matrix:
    weights = [weight for sublist in alternative_weights for weight in sublist]
    all_weights.append(weights)

# Transpose the list of lists to get weights for each alternative
alternatives_weights = list(map(list, zip(*all_weights)))

print(alternatives_weights)
