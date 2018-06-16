 for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (img[i][j] == r).all() or (img[i][j] == g).all() or (img[i][j] == b).all():
            img[i][j] = r