""" class to hold single frame
"""

import matplotlib.pyplot as plt

class FrameClass(object):
    """
    """
    def __init__(self, dat):

        if isinstance(dat, dict):
            # dict holding path to frame
            self.Path = dat["path"]
        elif isinstance(dat, str):
            # path to frame
            self.Path = dat

        self.Image = None
        self.ImageGray = None
    

    ###### TO IMAGE
    def image_extract(self):
        """ extract image
        OUT:
        - self.Image, self.ImageGray
        """
        
        import cv2
        img = cv2.imread(self.Path)
        assert img is not None, f"cant find {self.Path}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.Image = img
        self.ImageGray = gray


    ####### PLOTTING
    def plot_image(self):
        """ Plots both color and grayscale.
        extracts images first if not already done so
        """

        if self.Image is None or self.ImageGray is None:
            self.image_extract()
            
        fig, axes = plt.subplots(1,2, figsize=(40, 20))
        
        axes[0].imshow(self.Image, interpolation="bicubic")
        axes[1].imshow(self.ImageGray, cmap = 'gray', interpolation = 'bicubic')
        
        return fig

