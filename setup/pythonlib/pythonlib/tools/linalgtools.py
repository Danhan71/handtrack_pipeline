""" lin algebra stuff"""
import numpy as np
import matplotlib.pyplot as plt


def pca(X, centerize=True, ver="eig", ploton=False):
    """ 
    perform pca 
    - X, shape DxN, where N is datapts, and D is simensioanltiy
    - centerize, whether to demean data (i.e. each row, subtract )
    - ver, what method to use
    """

    if centerize:
        # - demean
        means = np.mean(X, axis=1)[:, None]
        X = X - means

    if ver=="svd":
        u, s, v = np.linalg.svd(X)
        w = s**2/(nunits-1)
        
    elif ver=="eig":
        Xcov = np.cov(X)
        w, u = np.linalg.eig(Xcov)

        # - plot
        if False:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            axes[0].imshow(Xcov);
            axes[1].hist(Xcov[:]);
            axes[1].set_title("elements in cov mat")

            # - plot 
            fig, axes = plt.subplots(1, 2, figsize=(15,5))

            axes[0].plot(w, '-ok')
            wsum = np.cumsum(w)/np.sum(w)
            axes[0].plot(wsum, '-or')
            axes[0].set_title('eigenvals')
            axes[1].imshow(v)
            axes[1].set_title('eigen vects')
            axes[1].set_xlabel('vect')

    w = w/np.sum(w)

    if ploton:
        import matplotlib.pyplot as plt
        # - plot 
        fig, axes = plt.subplots(1, 2, figsize=(15,5))

        axes[0].plot(w, '-ok')
        axes[1].plot(np.cumsum(w)/np.sum(w), '-or')
        axes[1].hlines(0.9, 0, len(w))

        axes[0].set_title('s vals')
        # axes[1].imshow(v)
        # axes[1].set_title('eigen vects')
        # axes[1].set_xlabel('vect')
    
    return w, u


