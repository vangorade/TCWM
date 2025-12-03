"""Test script for SlotContrastiveLoss integration."""

import torch
from modeling.losses import SlotContrastiveLoss


def test_slot_contrastive_loss():
    """Test the slot contrastive loss with synthetic data."""
    print("Testing SlotContrastiveLoss...")
    
    # Create synthetic slot data
    batch_size = 4
    sequence_length = 6
    num_slots = 7
    slot_dim = 64
    
    # Generate random slots
    slots = torch.randn(batch_size, sequence_length, num_slots, slot_dim)
    
    # Test with batch contrast enabled
    print(f"\nTest 1: Batch contrast enabled")
    loss_fn = SlotContrastiveLoss(temperature=0.1, batch_contrast=True)
    loss = loss_fn(slots)
    print(f"  Input shape: {slots.shape}")
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("  ✓ Passed")
    
    # Test with batch contrast disabled
    print(f"\nTest 2: Batch contrast disabled")
    loss_fn = SlotContrastiveLoss(temperature=0.1, batch_contrast=False)
    loss = loss_fn(slots)
    print(f"  Input shape: {slots.shape}")
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("  ✓ Passed")
    
    # Test with different temperature
    print(f"\nTest 3: Different temperature (0.5)")
    loss_fn = SlotContrastiveLoss(temperature=0.5, batch_contrast=True)
    loss = loss_fn(slots)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("  ✓ Passed")
    
    # Test gradient flow
    print(f"\nTest 4: Gradient flow")
    slots_with_grad = torch.randn(batch_size, sequence_length, num_slots, slot_dim, requires_grad=True)
    loss_fn = SlotContrastiveLoss(temperature=0.1, batch_contrast=True)
    loss = loss_fn(slots_with_grad)
    loss.backward()
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Gradient shape: {slots_with_grad.grad.shape}")
    assert slots_with_grad.grad is not None, "Gradients should be computed"
    assert not torch.isnan(slots_with_grad.grad).any(), "Gradients should not contain NaN"
    print("  ✓ Passed")
    
    # Test with perfect temporal consistency (same slots across time)
    print(f"\nTest 5: Perfect temporal consistency")
    perfect_slots = torch.randn(batch_size, 1, num_slots, slot_dim).repeat(1, sequence_length, 1, 1)
    loss_fn = SlotContrastiveLoss(temperature=0.1, batch_contrast=False)
    loss = loss_fn(perfect_slots)
    print(f"  Loss value with perfect consistency: {loss.item():.4f}")
    print(f"  (Lower is better - slots are identical across time)")
    print("  ✓ Passed")
    
    # Test action conditioning
    print(f"\nTest 6: Action-conditioned loss")
    action_dim = 4
    actions = torch.randn(batch_size, sequence_length - 1, action_dim)
    loss_fn = SlotContrastiveLoss(
        temperature=0.1, 
        batch_contrast=False,
        action_conditioned=True,
        slot_dim=slot_dim,
        action_dim=action_dim
    )
    loss = loss_fn(slots, actions)
    print(f"  Input shapes: slots={slots.shape}, actions={actions.shape}")
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("  ✓ Passed")
    
    # Test gradient flow with action conditioning
    print(f"\nTest 7: Action-conditioned gradient flow")
    slots_with_grad = torch.randn(batch_size, sequence_length, num_slots, slot_dim, requires_grad=True)
    actions_with_grad = torch.randn(batch_size, sequence_length - 1, action_dim, requires_grad=True)
    loss_fn = SlotContrastiveLoss(
        temperature=0.1,
        batch_contrast=True,
        action_conditioned=True,
        slot_dim=slot_dim,
        action_dim=action_dim
    )
    loss = loss_fn(slots_with_grad, actions_with_grad)
    loss.backward()
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Slot gradient shape: {slots_with_grad.grad.shape}")
    print(f"  Action gradient shape: {actions_with_grad.grad.shape}")
    assert slots_with_grad.grad is not None, "Slot gradients should be computed"
    assert actions_with_grad.grad is not None, "Action gradients should be computed"
    assert not torch.isnan(slots_with_grad.grad).any(), "Slot gradients should not contain NaN"
    assert not torch.isnan(actions_with_grad.grad).any(), "Action gradients should not contain NaN"
    print("  ✓ Passed")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_slot_contrastive_loss()
