"""Loss functions for SOLD training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotContrastiveLoss(nn.Module):
    """Slot-to-slot temporal contrastive loss from SlotContrast (CVPR 2025).
    
    Enforces temporal consistency by maximizing similarity between the same slot
    across consecutive frames while minimizing similarity to other slots.
    
    Can optionally be action-conditioned to account for dynamics.
    
    Reference: "Temporally Consistent Object-Centric Learning by Contrasting Slots"
    https://arxiv.org/abs/2412.14295
    
    Args:
        temperature: Temperature parameter for contrastive loss scaling.
        batch_contrast: If True, contrast slots across batch dimension as well.
        action_conditioned: If True, condition slot matching on actions.
        slot_dim: Dimension of slot representations (required if action_conditioned=True).
        action_dim: Dimension of action vectors (required if action_conditioned=True).
    """
    
    def __init__(self, temperature: float = 0.1, batch_contrast: bool = True,
                 action_conditioned: bool = False, slot_dim: int = None, action_dim: int = None):
        super().__init__()
        self.temperature = temperature
        self.batch_contrast = batch_contrast
        self.action_conditioned = action_conditioned
        self.criterion = nn.CrossEntropyLoss()
        
        # Action conditioning network
        if action_conditioned:
            if slot_dim is None or action_dim is None:
                raise ValueError("slot_dim and action_dim must be provided when action_conditioned=True")
            # Project actions to slot space
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, slot_dim),
                nn.LayerNorm(slot_dim),
                nn.ReLU(),
                nn.Linear(slot_dim, slot_dim)
            )
        else:
            self.action_encoder = None
    
    def forward(self, slots: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        """Compute slot-to-slot contrastive loss.
        
        Args:
            slots: Slot representations of shape (batch_size, sequence_length, num_slots, slot_dim)
            actions: Action vectors of shape (batch_size, sequence_length-1, action_dim) [optional]
        
        Returns:
            Scalar loss value
        """
        # Validate action conditioning
        if self.action_conditioned and actions is None:
            raise ValueError("actions must be provided when action_conditioned=True")
        
        # Get consecutive frame pairs
        s1 = slots[:, :-1, :, :]  # Slots at time t: [B, T-1, K, D]
        s2 = slots[:, 1:, :, :]   # Slots at time t+1: [B, T-1, K, D]
        
        # Action-conditioned: add action influence to slots at time t
        if self.action_conditioned:
            # Encode actions: [B, T-1, action_dim] -> [B, T-1, slot_dim]
            action_embeddings = self.action_encoder(actions)
            # Broadcast to all slots and add: [B, T-1, K, D]
            s1 = s1 + action_embeddings.unsqueeze(2)
        
        # Normalize slots to unit sphere for cosine similarity
        s1 = F.normalize(s1, p=2.0, dim=-1)
        s2 = F.normalize(s2, p=2.0, dim=-1)
        
        # Optionally contrast across batch dimension
        if self.batch_contrast:
            # Split batch and concatenate along slot dimension: [B, T, K, D] -> [1, T, B*K, D]
            s1 = s1.split(1)  # List of [1, T-1, K, D]
            s2 = s2.split(1)
            s1 = torch.cat(s1, dim=-2)  # [1, T-1, B*K, D]
            s2 = torch.cat(s2, dim=-2)
        
        # Compute similarity matrix: [B, T-1, K, K]
        # Each slot at time t (+ action) should match with the same slot at time t+1 (diagonal)
        similarity = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        
        B, T, S, _ = similarity.shape
        similarity = similarity.reshape(B * T, S, S)
        
        # Target: identity matrix (diagonal elements are positive pairs)
        target = torch.eye(S, device=similarity.device).expand(B * T, S, S)
        
        # Cross-entropy loss treats this as multi-class classification
        # where each slot at time t should match its corresponding slot at time t+1
        loss = self.criterion(similarity, target)
        
        return loss
