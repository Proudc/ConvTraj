import numpy as np

def onehot_encode(traj_list_onedim, num_classes):
    onehot_list = []
    for traj in traj_list_onedim:
        one_hot = np.zeros((len(traj), num_classes))
        one_hot[np.arange(len(traj)), traj] = 1.0
        onehot_list.append(one_hot.tolist())
    return onehot_list

def image_encode(x_list, y_list, x_num_classes, y_num_classes):
    image_list = []
    for i in range(len(x_list)):
        image = np.zeros((x_num_classes, y_num_classes))
        for j in range(len(x_list[i])):
            image[x_list[i][j]][y_list[i][j]] = 1.0
        image_list.append(image.tolist())
    return image_list