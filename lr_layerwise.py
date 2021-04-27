from torch.optim.lr_scheduler import LambdaLR


# Bert layerwise learning rates for Bio_ClinicalBERT, NOT USED
def Bio_ClinicalBERT_layer_lr(layer_name: str, args):
    name_list = layer_name.split(sep='.')
    if name_list[0] == 'embeddings':
        return args.lr * args.lr_layer_decay ** 14
    elif name_list[0] == 'final':
        return args.lr * args.lr_layer_decay ** 1
    elif len(name_list) > 3 and name_list[2].isdigit():
        # Middle layers in format encoder.layer.0.attention.self.query.weight, uses the layer int to count level of decay
        return args.lr * args.lr_layer_decay ** (13 - int(name_list[2]))
    else:
        # In case something missed, return default lr
        return args.lr




## BELOW CLASS NOT USED, UNTESTED, ALREADY IMPLEMENTED IN TRANSFORMERS
class SlantedTriangularLR:
    """
    Class for implementing slanted triangular learning rates following Howard & Ruder (2018),
    https://arxiv.org/pdf/1801.06146.pdf.

    ...

    Attributes
    -----------
    train_iter : int
        the number of training iterations
    warm-up : float
        % of warm-up iterations, between 0 and 1
    ratio : float
        specifies how much smaller is the lowest learning compared to the max learning rate
    optimizer : reference to PyTorch optimizer
    last_epoch : int
        The index of last epoch
    
    Methods:
    --------
    scheduler_function(t)
        Function to be used inside PyTorch LambdaLR scheduler
    get_scheduler()
        Returns PyTorch LambdaLR scheduler that implements slanted triangular learning rates

    """

    def __init__(self, train_iter, warm_up, ratio, optimizer, last_epoch = -1):
        self.train_iter = train_iter
        self.warm_up = warm_up
        self.ratio = self.ratio
        self.optimizer = optimizer
        self.last_epoch = last_epoch
    
    def scheduler_function(self, t):
        """
        Function implements slanted triangular learning rates.

        Parameters
        -----------
        t : int
            running int for training iterations
        """
        cut = self.train_iter * self.warm_up
        p = t / cut if t < cut else 1 - ( (t - cut) / ( cut * (1 / self.warm_up - 1)))
        return (1 + p * (self.ratio - 1)) / self.ratio

    def get_scheduler(self):
        return LambdaLR(self.optimizer, self.scheduler_function, self.last_epoch)
