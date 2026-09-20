"""
WAFS-Net
Wavelet-Adaptive Frequency-Spatial Network for
Benign and Malignant Ovarian Tumor Classification

Architecture:
    DenseNet121       -> 1024-D
    EfficientNet-B0   -> 1280-D
    Haar Wavelet      -> 256-D
                         |
                         v
    Concatenation     -> 2560-D
                         |
                         v
    Adaptive Gate     -> 2560 -> 640 -> 2560
                         |
                         v
    Classification    -> 2560 -> 768 -> 256 -> 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import pywt
except ImportError:
    pywt = None


# ============================================================
# 1. Haar Wavelet Branch
# ============================================================

class HaarWaveletBranch(nn.Module):
    """
    Frequency-domain branch based on one-level Haar DWT.

    For each RGB channel, four wavelet sub-bands are obtained:
        LL, LH, HL, HH

    Global statistics are extracted from the sub-bands,
    producing a 12-D wavelet descriptor.

    The descriptor is projected as:
        12 -> 64 -> 128 -> 256
    """

    def __init__(self, out_dim=256):
        super().__init__()

        self.wavelet_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _haar_statistics_single(image):
        """
        Extract a 12-D Haar-wavelet descriptor from one image.

        Parameters
        ----------
        image : torch.Tensor
            Tensor with shape [3, H, W].

        Returns
        -------
        torch.Tensor
            12-dimensional wavelet descriptor.
        """

        if pywt is None:
            raise ImportError(
                "PyWavelets is required. "
                "Install it using: pip install PyWavelets"
            )

        image_np = image.detach().cpu().numpy()

        features = []

        for channel in image_np:

            LL, (LH, HL, HH) = pywt.dwt2(
                channel,
                "haar"
            )

            # One global statistic for each sub-band.
            features.extend(
                [
                    float(abs(LL).mean()),
                    float(abs(LH).mean()),
                    float(abs(HL).mean()),
                    float(abs(HH).mean()),
                ]
            )

        return torch.tensor(
            features,
            dtype=torch.float32
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch [B, 3, H, W].

        Returns
        -------
        torch.Tensor
            Frequency representation [B, 256].
        """

        device = x.device

        wavelet_features = []

        for image in x:
            descriptor = self._haar_statistics_single(
                image
            )
            wavelet_features.append(descriptor)

        wavelet_features = torch.stack(
            wavelet_features,
            dim=0
        ).to(device)

        wavelet_features = self.wavelet_mlp(
            wavelet_features
        )

        return wavelet_features


# ============================================================
# 2. DenseNet121 Spatial Encoder
# ============================================================

class DenseNetEncoder(nn.Module):
    """
    DenseNet121 spatial encoder.

    Output dimension:
        1024
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
        else:
            weights = None

        backbone = models.densenet121(
            weights=weights
        )

        self.features = backbone.features

        self.out_dim = 1024

    def forward(self, x):

        x = self.features(x)

        x = F.relu(
            x,
            inplace=True
        )

        x = F.adaptive_avg_pool2d(
            x,
            output_size=(1, 1)
        )

        x = torch.flatten(
            x,
            1
        )

        return x


# ============================================================
# 3. EfficientNet-B0 Spatial Encoder
# ============================================================

class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 spatial encoder.

    Output dimension:
        1280
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT
            )
        else:
            weights = None

        backbone = models.efficientnet_b0(
            weights=weights
        )

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.out_dim = 1280

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        return x


# ============================================================
# 4. Dual Spatial Encoder
# ============================================================

class DualSpatialEncoder(nn.Module):
    """
    Dual spatial representation:

        DenseNet121       -> 1024-D
        EfficientNet-B0   -> 1280-D
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.densenet = DenseNetEncoder(
            pretrained=pretrained
        )

        self.efficientnet = EfficientNetEncoder(
            pretrained=pretrained
        )

    def forward(self, x):

        dense_features = self.densenet(x)

        efficient_features = self.efficientnet(x)

        return (
            dense_features,
            efficient_features
        )


# ============================================================
# 5. Adaptive Frequency-Spatial Fusion
# ============================================================

class AdaptiveFrequencySpatialFusion(nn.Module):
    """
    Adaptive frequency-spatial fusion.

    Input dimensions:
        DenseNet121      : 1024
        EfficientNet-B0  : 1280
        Wavelet          : 256

    Total:
        1024 + 1280 + 256 = 2560

    Adaptive gate:
        2560 -> 640 -> 2560
    """

    def __init__(
        self,
        dense_dim=1024,
        efficient_dim=1280,
        wavelet_dim=256
    ):
        super().__init__()

        self.input_dim = (
            dense_dim
            + efficient_dim
            + wavelet_dim
        )

        assert self.input_dim == 2560

        self.gate = nn.Sequential(

            nn.Linear(
                self.input_dim,
                640
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                640,
                self.input_dim
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        dense_features,
        efficient_features,
        wavelet_features
    ):

        fused_features = torch.cat(
            [
                dense_features,
                efficient_features,
                wavelet_features
            ],
            dim=1
        )

        gate_weights = self.gate(
            fused_features
        )

        gated_features = (
            fused_features
            * gate_weights
        )

        return gated_features


# ============================================================
# 6. Classification Head
# ============================================================

class BinaryClassificationHead(nn.Module):
    """
    Binary classification head.

    Architecture:
        2560 -> 768 -> 256 -> 2
    """

    def __init__(
        self,
        input_dim=2560,
        num_classes=2,
        dropout=0.5
    ):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dim,
                768
            ),

            nn.BatchNorm1d(768),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                768,
                256
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        return self.classifier(x)


# ============================================================
# 7. Complete WAFS-Net
# ============================================================

class DualWaveletBinaryModel(nn.Module):
    """
    Complete WAFS-Net model.

    Spatial branches:
        DenseNet121       -> 1024-D
        EfficientNet-B0   -> 1280-D

    Frequency branch:
        Haar DWT          -> 256-D

    Fusion:
        1024 + 1280 + 256
        = 2560-D

    Adaptive gate:
        2560 -> 640 -> 2560

    Classification:
        2560 -> 768 -> 256 -> 2
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.5
    ):
        super().__init__()

        # --------------------------------------------
        # Spatial feature extraction
        # --------------------------------------------

        self.spatial_encoder = DualSpatialEncoder(
            pretrained=pretrained
        )

        # --------------------------------------------
        # Frequency feature extraction
        # --------------------------------------------

        self.wavelet_branch = HaarWaveletBranch(
            out_dim=256
        )

        # --------------------------------------------
        # Adaptive fusion
        # --------------------------------------------

        self.fusion = (
            AdaptiveFrequencySpatialFusion(
                dense_dim=1024,
                efficient_dim=1280,
                wavelet_dim=256
            )
        )

        # --------------------------------------------
        # Classification
        # --------------------------------------------

        self.classifier = (
            BinaryClassificationHead(
                input_dim=2560,
                num_classes=num_classes,
                dropout=dropout
            )
        )

    def forward(
        self,
        x,
        return_features=False
    ):

        # Spatial representations
        dense_features, efficient_features = (
            self.spatial_encoder(x)
        )

        # Frequency representation
        wavelet_features = (
            self.wavelet_branch(x)
        )

        # Adaptive frequency-spatial fusion
        fused_features = self.fusion(
            dense_features,
            efficient_features,
            wavelet_features
        )

        # Final prediction
        logits = self.classifier(
            fused_features
        )

        if return_features:

            return {
                "logits": logits,
                "dense_features": dense_features,
                "efficient_features": efficient_features,
                "wavelet_features": wavelet_features,
                "fused_features": fused_features
            }

        return logits


# ============================================================
# 8. Alias used in the manuscript/repository
# ============================================================

class WAFSNet(DualWaveletBinaryModel):
    """
    Alias for DualWaveletBinaryModel.
    """

    pass


# ============================================================
# 9. Model Builder
# ============================================================

def build_wafs_net(
    num_classes=2,
    pretrained=True,
    dropout=0.5
):
    """
    Build WAFS-Net.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether ImageNet-pretrained spatial
        backbones are used.

    dropout : float
        Dropout probability in the
        classification head.

    Returns
    -------
    WAFSNet
        Initialized model.
    """

    model = WAFSNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    return model


# ============================================================
# 10. Basic Architecture Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_wafs_net(
        num_classes=2,
        pretrained=False
    ).to(device)

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    model.eval()

    with torch.no_grad():

        outputs = model(
            dummy_input,
            return_features=True
        )

    print(
        "Input:",
        dummy_input.shape
    )

    print(
        "DenseNet121 feature:",
        outputs[
            "dense_features"
        ].shape
    )

    print(
        "EfficientNet-B0 feature:",
        outputs[
            "efficient_features"
        ].shape
    )

    print(
        "Wavelet feature:",
        outputs[
            "wavelet_features"
        ].shape
    )

    print(
        "Fused feature:",
        outputs[
            "fused_features"
        ].shape
    )

    print(
        "Output:",
        outputs[
            "logits"
        ].shape
    )
