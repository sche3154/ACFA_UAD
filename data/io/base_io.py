from abc import ABC, abstractmethod

class BaseIO(ABC):

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.opt = opt
        self.root = opt.dataroot
        self.isTrain = opt.isTrain

    #---------------------------------------------#
    #               load_data                     #
    #---------------------------------------------#
    @abstractmethod
    def load_sample(self, index):

        pass

    #---------------------------------------------#
    #               save_data                     #
    #---------------------------------------------#
    @abstractmethod
    def save_sample(self, sample):

        pass



    