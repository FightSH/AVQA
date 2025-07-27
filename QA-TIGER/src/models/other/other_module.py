import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAdjuster(nn.Module):
    def __init__(self):
        super(FeatureAdjuster, self).__init__()
        self.relu_activation = nn.ReLU(inplace=False)

    def forward(self, feature_a, feature_b):
        cosine_similarity = F.cosine_similarity(feature_b, feature_a, dim=2)
        cosine_similarity = cosine_similarity.unsqueeze(2)

        feature_a = feature_a+feature_b*cosine_similarity

        feature_b = feature_b+feature_a*cosine_similarity

        feature_a = self.relu_activation(feature_a)
        feature_b = self.relu_activation(feature_b)
        return feature_a,feature_b


# if __name__  == "__main__":
#     # Example usage
#     feature_a = torch.randn(10,60, 512)  # Example feature A
#     feature_b = torch.randn(10, 60,512)  # Example feature B
#
#     adjuster = FeatureAdjuster()
#     adjusted_feature = adjuster(feature_a, feature_b)
#     print(adjusted_feature.shape)  # Should be the same shape as feature_a and feature_b