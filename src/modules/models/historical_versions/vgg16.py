import torch, math
import torch.nn as nn
from rich import print
from torch import stack
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import BatchNorm2d
from src.utils.general import banner
from src.modules.layers.flex import Flex2D
from src.modules.models.utils import DimensionTracer


class VGG16(nn.Module):
    def __init__(self, config, verbose=True):
        super().__init__()

        # -------- init configs --------
        self.config = config
        self.default_dense_size = 4096  # the value will be used if the network is not too small or deep

        # -------- IMPORTANT --------
        dimension_tracer = DimensionTracer(config.get("in_dimensions"))

        # -------- nesting conv2d --------
        class Conv2d(Flex2D if self.config.get("use_flex") else nn.Conv2d):
            def __init__(self, *args, **kwargs):
                # -------- conditional init --------
                if isinstance(self, Flex2D):
                    kwargs["config"] = config
                super().__init__(*args, **kwargs)

                # -------- figure out all the in and out dimensions --------
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

                if isinstance(self, Flex2D):
                    self.init_dimension_dependent_modules()

        # -------- nesting maxpool2d --------
        class MaxPool2d(nn.MaxPool2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # -------- figure out all the in and out dimensions --------
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

            def forward(self, x):  # for hessian computation we need indices
                self.output, self.indices = F.max_pool2d(
                    x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, return_indices=True
                )
                return self.output

        # -------- nesting relu -------- in case we need to easily change the activation function to something else
        class ReLU(nn.ReLU):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        # -------- end nesting --------
        self.features = nn.Sequential(
            # Block 1
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        # -------- layer mapping --------
        feature_flatten_size = math.prod(dimension_tracer.calculate_dimension())

        if feature_flatten_size <= self.default_dense_size:
            self.default_dense_size = feature_flatten_size
            if verbose:
                banner("WARNING")
                print(
                    f"[bold red]WARNING:[/bold red] Feature size is less than 4096, shrinking dense size: -> {feature_flatten_size}"
                )
                banner("")

        # --------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_flatten_size, out_features=self.default_dense_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.get("dropout", 0.2)),
            nn.Linear(in_features=self.default_dense_size, out_features=self.default_dense_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.get("dropout", 0.2)),
            nn.Linear(in_features=self.default_dense_size, out_features=config.get("num_classes")),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # flatten feature tensor
        x = self.classifier(x)
        return x

    def update_tau(self, tau_value):
        """
        Set the tau value for all Flex2D layers for sigmoid smoothing during training.
        """
        assert self.config.get("use_flex"), "This model does not use Flex2D layers."
        for module in self.modules():
            if isinstance(module, Flex2D):
                module.tau = tau_value

    def check_tau(self):
        """
        check the tau value for all Flex2D layers for Gumbel-Softmax during training.
        """
        assert self.config.get("use_flex"), "This model does not use Flex2D layers."
        for module in self.modules():
            if isinstance(module, Flex2D):
                print(f"tau: {module.tau}")
                return module.tau

    def check_binariness(self):
        """
        checking how binary each flex2d mask is
        this outputs a list of length num of flex2d used in the network
        each is a float between 0 and 1
        higher the more binary
        """
        assert self.config.get("use_flex"), "This model does not use Flex2D layers for checking binariness."
        binariness_values = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                binariness_values.append(module.homogeneity)
        return binariness_values

    def _get_cp_id_matrices(self):
        assert self.config.get("use_flex"), "This model does not use Flex2D layers for checking conv ratio."
        cp_identity_matrices = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                cp_identity_matrix = (module.cp_identity_matrix >= 0.5).float()
                cp_identity_matrices.append(cp_identity_matrix)
        return cp_identity_matrices

    def check_conv_ratio(self):
        """
        check the ratio of conv in each flex2d mask
        defined as the ratio of values above 0.5
        this outputs a list of length num of flex2d used in the network
        each is a float between 0 and 1
        higher the more conv
        """
        assert self.config.get("use_flex"), "This model does not use Flex2D layers for checking conv ratio."
        conv_ratios = []
        for module in self.modules():
            if isinstance(module, Flex2D):
                conv_ratios.append(module.conv_ratio)
        return conv_ratios


if __name__ == "__main__":
    config = {"use_flex": False, "in_dimensions": (3, 32, 32), "num_classes": 10}
    model = VGG16(config)
    summary(model, (3, 32, 32))
