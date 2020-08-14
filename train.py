"""
Michael Patel
August 2020

Project description:

File description:

conda activate Anomaly-Detection
"""
################################################################################
# Imports
from model import build_autoencoder
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

    print(f'Anomaly label: {ANOMALY_LABEL}')
    print(f'Anomaly category: {int2class[ANOMALY_LABEL]}')

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
        if train_labels[i] != ANOMALY_LABEL:
            x_images.append(train_images[i])
            x_labels.append(train_labels[i])

        else:  # label = ANOMALY_LABEL
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

    train_images = train_images.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    test_images = test_images.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    print(f'Shape of train images: {train_images.shape}')
    print(f'Shape of train labels: {train_labels.shape}')
    print(f'Shape of test images: {test_images.shape}')
    print(f'Shape of test labels: {test_labels.shape}')

    # ----- MODEL ----- #
    model = build_autoencoder()

    model.compile(
        loss=tf.keras.losses.mse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()

    # ----- TRAIN ----- #
    history = model.fit(
        x=train_images,
        y=train_images,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )

    # plot training loss
    plt.plot(history.history["loss"], label="loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(os.getcwd(), "training"))

    # save model
    model.save(SAVE_DIR)

    # ----- PREDICT ----- #
    # calculate threshold
    prediction = model.predict(train_images)
    train_mse_loss = np.mean(np.power((prediction-train_images), 2))

    threshold = np.max(train_mse_loss)  # single value threshold

    # calculate test MSE loss
    test_prediction = model.predict(test_images)
    test_diff = test_prediction - test_images
    test_mse_loss = []
    for i in range(len(test_diff)):
        mse = np.mean(np.power(test_diff[i], 2))
        test_mse_loss.append(mse)

    # plot test loss
    plt.close()
    plt.hist(test_mse_loss, bins=100)
    plt.title("Test Loss")
    plt.xlabel("Test Loss")
    plt.ylabel("Number of samples")
    plt.grid()
    plt.savefig(os.path.join(os.getcwd(), "test_histogram"))

    # detect anomalies
    # actual indices
    idx = []
    for i in range(len(test_labels)):
        if test_labels[i] == ANOMALY_LABEL:
            idx.append(i)

    print(f'Actual indices of label=9: {idx}')
    print(f'Actual number of label=9: {len(idx)}')
    print()

    # predicted indices
    pred_idx = []
    for i in range(len(test_mse_loss)):
        if test_mse_loss[i] > threshold:
            pred_idx.append(i)

    print(f'Predicted indices of label=9: {pred_idx}')
    print(f'Number of anomalies: {len(pred_idx)}')
