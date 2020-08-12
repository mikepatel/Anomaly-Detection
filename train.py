"""
Michael Patel
August 2020

Project description:

File description:

conda activate Anomaly-Detection
"""
################################################################################
# Imports
from parameters import *


################################################################################
# Main
if __name__ == "__main__":
    # print TF version
    print(f'TF version: {tf.__version__}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    """
    print(f'Shape of train images: {train_images.shape}')
    print(f'Shape of train labels: {train_labels.shape}')
    print(f'Shape of test images: {test_images.shape}')
    print(f'Shape of test labels: {test_labels.shape}')
    """

    # class labels
    categories = {
        "t-shirt": 0,
        "trouser": 0,
        "pullover": 0,
        "dress": 0,
        "coat": 0,
        "sandal": 0,
        "shirt": 0,
        "sneaker": 0,
        "bag": 0,
        "ankle boot": 0
    }

    class2int = {u: i for i, u in enumerate(categories)}
    int2class = {i: u for i, u in enumerate(categories)}

    #print(class2int)

    # get count of each category, 6k of each
    for i in range(len(train_labels)):
        item = int2class[train_labels[i]]
        categories[item] += 1

    #print(categories)  # updated counts

    # intentionally introduce class imbalance
    # 0.5% --> 6000 * 0.005 = 30
    x_images = []
    x_labels = []
    a_images = []
    a_labels = []
    for i in range(len(train_images)):
        if train_labels[i] != 9:  # label != 9
            x_images.append(train_images[i])
            x_labels.append(train_labels[i])

        else:  # label = 9
            a_images.append(train_images[i])
            a_labels.append(train_labels[i])

    """
    print(len(x_images))
    print(len(x_labels))

    print(len(a_images[:30]))
    print(len(a_labels[:30]))
    """

    train_images = x_images + a_images[:30]
    train_images = np.array(train_images)
    train_labels = x_labels + a_labels[:30]
    train_labels = np.array(train_labels)

    print(f'Shape of train images: {train_images.shape}')
    print(f'Shape of train labels: {train_labels.shape}')
    print(f'Shape of test images: {test_images.shape}')
    print(f'Shape of test labels: {test_labels.shape}')

    # ----- MODEL ----- #

    # ----- TRAIN ----- #

    # ----- PREDICT ----- #
