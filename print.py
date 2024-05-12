import pickle

import matplotlib.pyplot as plt

def group_ones_with_zeroes(lst):
    groups = [[]]
    for num in lst:
        groups[-1].append(num)
        if num == 0:
            groups.append([])
    return [group + [0] * (10 - len(group)) for group in groups if group]


def find_group_starts(lst):
    return [i for i, (a, b) in enumerate(zip(lst, lst[1:]), 1) if a == 1 and b == 0]


def group_starting_points_dynamic(start_points):
    if len(start_points) == 0:
        return []

    # Calculate the distances between consecutive starting points
    distances = [start_points[i + 1] - start_points[i] for i in range(len(start_points) - 1)]

    # Calculate the average distance
    avg_distance = sum(distances) / len(distances)

    # Set the threshold as a multiple of the average distance
    threshold = 1.6 * avg_distance  # Adjust the multiplier as needed

    # Group starting points based on the dynamic threshold
    grouped_start_points = []
    current_group = []

    for i, start_point in enumerate(start_points):
        if i == 0:
            current_group.append(start_point)
        else:
            if start_point - current_group[-1] <= threshold:
                current_group.append(start_point)
            else:
                grouped_start_points.append(current_group)
                current_group = [start_point]

    grouped_start_points.append(current_group)  # Append the last group

    return grouped_start_points

with open('cluster_list.pkl', 'rb') as file:
    # Use pickle to load the list from the file
    loaded_data = pickle.load(file)

print("List loaded from file:", len(loaded_data))

grouped_data = group_ones_with_zeroes(loaded_data)
group_starts = find_group_starts(loaded_data)
grouped_start_points_dynamic = group_starting_points_dynamic(group_starts)

fps = 50

print(len(grouped_start_points_dynamic))
print(grouped_start_points_dynamic)

Time0_5 = (grouped_start_points_dynamic[0][0] * 1 / fps)
Time5_15 = (grouped_start_points_dynamic[1][0] * 1 / fps)-(grouped_start_points_dynamic[0][0] * 1 / fps)
Time15_25 = (grouped_start_points_dynamic[2][0] * 1 / fps)-(grouped_start_points_dynamic[1][0] * 1 / fps)
Time25_35 = (grouped_start_points_dynamic[3][0] * 1 / fps)-(grouped_start_points_dynamic[2][0] * 1 / fps)
Time35_45 = (grouped_start_points_dynamic[4][0] * 1 / fps)-(grouped_start_points_dynamic[3][0] * 1 / fps)
Time45_55 = (grouped_start_points_dynamic[5][0] * 1 / fps)-(grouped_start_points_dynamic[4][0] * 1 / fps)
Time55_65 = (grouped_start_points_dynamic[6][0] * 1 / fps)-(grouped_start_points_dynamic[5][0] * 1 / fps)
Time65_75 = (grouped_start_points_dynamic[7][0] * 1 / fps)-(grouped_start_points_dynamic[6][0] * 1 / fps)
Time75_85 = (grouped_start_points_dynamic[8][0] * 1 / fps)-(grouped_start_points_dynamic[7][0] * 1 / fps)
# Time85_95 = (grouped_start_points_dynamic[9][0] * 1 / fps)-(grouped_start_points_dynamic[8][0] * 1 / fps)
total_sum = sum((grouped_start_points_dynamic[i+1][0] - grouped_start_points_dynamic[i][0]) / fps for i in range(8))+Time0_5

print(f'0m-5m -> {Time0_5}s')
print(f'5m-15m -> {Time5_15}s')
print(f'15m-25m -> {Time15_25}s')
print(f'25m-35m -> {Time25_35}s')
print(f'35m-45m -> {Time35_45}s')
print(f'45m-55m -> {Time45_55}s')
print(f'55-65m -> {Time55_65}s')
print(f'65-75m -> {Time65_75}s')
print(f'75-85m -> {Time75_85}s')
# print(f'75-95m -> {Time85_95}s')

minutes, seconds = divmod(int(total_sum+3.98), 60)
milliseconds = int((total_sum - int(total_sum)) * 1000)
Time_25 = sum([Time0_5, Time5_15, Time15_25])
print(f"First 25m time -> {grouped_start_points_dynamic[2][0] * 1 / 50}s")

print(f"{minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")

average_speed = 95 / total_sum
print(average_speed)

total_10 = ""


plt.plot(loaded_data)
plt.show()