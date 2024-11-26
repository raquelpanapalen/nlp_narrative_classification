import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMultiLabelLoss(nn.Module):
    def __init__(
        self,
        num_level2_classes={"CC": 3, "URW": 2},
        level1_weight=1.0,
        level2_weight=1.0,
        level3_weight=1.0,
        dynamic_weighting=False,
    ):
        """
        Custom hierarchical multi-label loss function

        Parameters:
        - num_level2_classes: Dict specifying number of classes for each branch
        - level1_weight: Base weight for level 1 loss
        - level2_weight: Base weight for level 2 loss
        - level3_weight: Base weight for level 3 loss
        - dynamic_weighting: If True, dynamically adjust weights during training
        """
        super(HierarchicalMultiLabelLoss, self).__init__()

        self.num_level2_classes = num_level2_classes
        self.level1_weight = level1_weight
        self.level2_weight = level2_weight
        self.level3_weight = level3_weight
        self.dynamic_weighting = dynamic_weighting

        # If dynamic weighting is enabled, create buffers to track loss moving averages
        if dynamic_weighting:
            self.register_buffer("level1_ma", torch.tensor(1.0))
            self.register_buffer("level2_ma", torch.tensor(1.0))
            self.register_buffer("level3_ma", torch.tensor(1.0))
            self.alpha = 0.9  # moving average decay factor

    def _compute_dynamic_weights(self, level1_loss, level2_loss, level3_loss):
        """
        Dynamically compute weights based on moving averages of losses
        """
        # Update moving averages
        self.level1_ma = (
            self.alpha * self.level1_ma + (1 - self.alpha) * level1_loss.detach()
        )
        self.level2_ma = (
            self.alpha * self.level2_ma + (1 - self.alpha) * level2_loss.detach()
        )
        self.level3_ma = (
            self.alpha * self.level3_ma + (1 - self.alpha) * level3_loss.detach()
        )

        # Calculate relative scales
        total_ma = self.level1_ma + self.level2_ma + self.level3_ma

        return {
            "level1_weight": total_ma / (3 * self.level1_ma),
            "level2_weight": total_ma / (3 * self.level2_ma),
            "level3_weight": total_ma / (3 * self.level3_ma),
        }

    def forward(
        self,
        level1_probs,
        level2_probs,
        level3_probs,
        level1_labels,
        level2_labels,
        level3_labels,
    ):
        """
        Compute hierarchical multi-label loss

        Params:
        - level1_probs: Probabilities for level 1 [B, num_classes]
        - level2_probs: Dict of probabilities for level 2 branches
        - level3_probs: Dict of probabilities for level 3 fine-grained labels
        - level1_labels: Ground truth labels for level 1 [B, num_classes]
        - level2_labels: Dict of ground truth labels for level 2 branches
        - level3_labels: Dict of ground truth labels for level 3 fine-grained labels
        """
        # Level 1 loss (Binary Cross Entropy for multi-label)
        level1_loss = F.binary_cross_entropy(level1_probs, level1_labels)

        # Level 2 loss (Binary Cross Entropy for multi-label)
        batch_size = level1_labels.size(0)
        level2_loss = 0.0

        # Compute level 2 loss for both CC and URW branches
        for branch in ["CC", "URW"]:
            branch_idx = 0 if branch == "CC" else 1
            active_samples = level1_labels[:, branch_idx] > 0.5

            if active_samples.any():
                level2_loss += F.binary_cross_entropy(
                    level2_probs[branch][active_samples],
                    level2_labels[branch][active_samples],
                )

        # Average over branches if any loss was computed
        level2_loss = level2_loss / 2 if level2_loss > 0 else 0.0

        # Level 3 loss (Binary Cross Entropy for multi-label)
        level3_loss = 0.0
        count = 0

        # Calculate level 3 loss for both CC and URW branches
        for branch in ["CC", "URW"]:
            branch_idx = 0 if branch == "CC" else 1
            active_samples = level1_labels[:, branch_idx] > 0.5

            if active_samples.any():
                for i in range(batch_size):
                    if active_samples[i]:
                        # Get active level 2 labels for this branch
                        active_l2_labels = torch.nonzero(
                            level2_labels[branch][i]
                        ).squeeze(1)

                        # Compute loss for active level 3 branches
                        for l2_idx in active_l2_labels:
                            label_key = f"{branch}{l2_idx.item()+1}"
                            if label_key in level3_probs:
                                level3_loss += F.binary_cross_entropy(
                                    level3_probs[label_key][i],
                                    level3_labels[label_key][i],
                                )
                                count += 1

        # Average level 3 loss
        level3_loss = level3_loss / count if count > 0 else 0.0

        # Dynamic or fixed weighting
        if self.dynamic_weighting:
            weights = self._compute_dynamic_weights(
                level1_loss, level2_loss, level3_loss
            )
            level1_weight = weights["level1_weight"]
            level2_weight = weights["level2_weight"]
            level3_weight = weights["level3_weight"]
        else:
            level1_weight = self.level1_weight
            level2_weight = self.level2_weight
            level3_weight = self.level3_weight

        # Combine losses with weighting
        total_loss = (
            level1_weight * level1_loss
            + level2_weight * level2_loss
            + level3_weight * level3_loss
        )

        return total_loss


# Example usage:
"""
# Create loss function
loss_fn = HierarchicalMultiLabelLoss(
    num_level2_classes={'CC': 3, 'URW': 2},
    dynamic_weighting=True
)

# In training loop
loss = loss_fn(
    level1_probs=model_output[0], 
    level2_probs=model_output[1], 
    level3_probs=model_output[2], 
    level1_labels=batch['level1_labels'], 
    level2_labels=batch['level2_labels'],
    level3_labels=batch['level3_labels']
)
loss.backward()
"""
