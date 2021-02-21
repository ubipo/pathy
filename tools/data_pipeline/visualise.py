import matplotlib.pyplot as plt


def show_two(title1, img1, title2, img2):
    plt.figure(figsize=(11,8), dpi=100)
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.imshow(img1)

    plt.subplot(1,2,2)
    plt.title(title2)
    plt.imshow(img2)

# def dataset_to_numpy(dataset, n):
#     return next(iter(visualisation_dataset))
