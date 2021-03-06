from typing import Mapping, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


# https://stackoverflow.com/a/312464/7120579
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def show_two(title1, img1, title2, img2):
    plt.figure(figsize=(11,8), dpi=100)
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.imshow(img1)

    plt.subplot(1,2,2)
    plt.title(title2)
    plt.imshow(img2)

def show(images: Mapping[str, np.ndarray]):
    plt.figure(figsize=(11,8), dpi=100)
    rows = list(chunks(list(images.items()), 3))
    nmbro_rows = len(rows)
    max_row_length = max((len(r) for r in rows))

    for y, row in enumerate(rows):
        for x, (name, image) in enumerate(row):
            plt.subplot(nmbro_rows, max_row_length, y + x + 1)
            plt.title(name)
            plt.imshow(image)

def show_rows(columns: Mapping[str, List[np.ndarray]]):
    plt.figure(figsize=(11,8), dpi=100)
    nmbro_columns = len(columns)
    max_column_length = max((len(c) for c in columns.values()))
    f, axarr = plt.subplots(max_column_length, nmbro_columns)

    for y, (name, column) in enumerate(columns.items()):
        for x, image in enumerate(column):
            axarr[x, y].subplot(max_column_length, nmbro_columns)
            if y == 0:
                axarr[x, y].title(name)
            axarr[x, y].imshow(image)

# def dataset_to_numpy(dataset, n):
#     return next(iter(visualisation_dataset))
