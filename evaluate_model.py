import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from model import RallyPredictor

def evaluate_model(model_path, test_loader, device=None, tolerance=0.25):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RallyPredictor().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['rally_predictor_state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    action_classes = {v: k for k, v in test_loader.dataset.action_classes.items()}
    group_classes = {v: k for k, v in test_loader.dataset.group_classes.items()}
    
    group_to_rally = {
        'l-pass': 0, 'r-pass': 1, 'l_set': 0, 'r_set': 1,
        'l-spike': 0, 'r_spike': 1, 'l_winpoint': 0, 'r_winpoint': 1
    }

    all_person_preds, all_person_targets = [], []
    all_group_preds, all_group_targets = [], []
    all_rally_probs, all_heuristic_probs = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, bboxes, person_actions, group_activity = batch
            images, bboxes = images.to(device), bboxes.to(device)

            _, action_logits, group_logits, rally_probs, heuristic_probs = model(images, bboxes)
            action_preds = action_logits.argmax(dim=2).view(-1).cpu().numpy()
            group_preds = group_logits.argmax(dim=1).cpu().numpy()
            person_targets = person_actions.view(-1).cpu().numpy()
            group_targets = group_activity.cpu().numpy()

            all_person_preds.extend(action_preds)
            all_person_targets.extend(person_targets)
            all_group_preds.extend(group_preds)
            all_group_targets.extend(group_targets)
            all_rally_probs.extend(rally_probs.cpu().numpy())
            all_heuristic_probs.extend(heuristic_probs.cpu().numpy())

            if batch_idx == 0:
                print("Sample rally_probs:", rally_probs[:5].cpu().numpy())
                print("Sample heuristic probs:", heuristic_probs[:5].cpu().numpy())

    all_person_preds = np.array(all_person_preds)
    all_person_targets = np.array(all_person_targets)
    all_group_preds = np.array(all_group_preds)
    all_group_targets = np.array(all_group_targets)
    all_rally_probs = np.array(all_rally_probs)
    all_heuristic_probs = np.array(all_heuristic_probs)

    print(f"rally_probs shape: {all_rally_probs.shape}, sum check: {np.sum(all_rally_probs, axis=1)[:5]}")
    print(f"heuristic probs shape: {all_heuristic_probs.shape}, sum check: {np.sum(all_heuristic_probs, axis=1)[:5]}")

    person_acc = (all_person_preds == all_person_targets).mean() * 100
    group_acc = (all_group_preds == all_group_targets).mean() * 100

    person_class_accs = {}
    for class_idx in range(len(action_classes)):
        class_mask = all_person_targets == class_idx
        if np.sum(class_mask) > 0:
            acc = (all_person_preds[class_mask] == class_idx).mean() * 100
            person_class_accs[action_classes[class_idx]] = acc

    group_class_accs = {}
    for class_idx in range(len(group_classes)):
        class_mask = all_group_targets == class_idx
        if np.sum(class_mask) > 0:
            acc = (all_group_preds[class_mask] == class_idx).mean() * 100
            group_class_accs[group_classes[class_idx]] = acc

    avg_person_class_acc = np.mean(list(person_class_accs.values())) if person_class_accs else 0.0
    avg_group_class_acc = np.mean(list(group_class_accs.values())) if group_class_accs else 0.0

    # Filtered Soft Consistency
    total_clips = len(all_rally_probs)
    filtered_indices = [i for i in range(total_clips) if np.max(all_heuristic_probs[i]) < 1.0]
    filtered_rally_probs = all_rally_probs[filtered_indices]
    filtered_heuristic_probs = all_heuristic_probs[filtered_indices]
    filtered_total = len(filtered_indices)
    consistency_matches = 0
    for i in range(filtered_total):
        max_diff = np.max(np.abs(filtered_rally_probs[i] - filtered_heuristic_probs[i]))
        if max_diff < tolerance:
            consistency_matches += 1
    soft_consistency = (consistency_matches / filtered_total) * 100 if filtered_total > 0 else 0.0

    # Group-Aligned Rally Consistency
    rally_winner = np.argmax(all_rally_probs, axis=1)
    group_implied_winner = np.array([group_to_rally[group_classes[t]] for t in all_group_targets])
    rally_group_matches = (rally_winner == group_implied_winner).sum()
    group_aligned_consistency = (rally_group_matches / total_clips) * 100 if total_clips > 0 else 0.0

    # Rally Probability Averages
    heuristic_avg = np.mean(all_heuristic_probs, axis=0)
    nn_avg = np.mean(all_rally_probs, axis=0)
    gt_avg = np.mean(np.eye(2)[group_implied_winner], axis=0)

    print(f"Evaluation on Test Set ({len(all_group_targets)} samples):")
    print(f"Person Action - Accuracy: {person_acc:.2f}%, Per-Class Accuracy: {avg_person_class_acc:.2f}%")
    print(f"Group Activity - Accuracy: {group_acc:.2f}%, Per-Class Accuracy: {avg_group_class_acc:.2f}%")
    print(f"\nRally Prediction - Filtered Soft Consistency Score: {soft_consistency:.2f}%")
    print(f"  (Percentage of {filtered_total} clips (excluding heuristic [0, 1] or [1, 0]) where max difference is less than {tolerance})")
    print(f"Rally Prediction - Group-Aligned Consistency Score: {group_aligned_consistency:.2f}%")
    print(f"  (Percentage of clips where rally winner matches group activity-implied winner)")
    print(f"\nRally Probability Averages:")
    print(f"Heuristic - Left: {heuristic_avg[0]:.3f}, Right: {heuristic_avg[1]:.3f}")
    print(f"Neural Net - Left: {nn_avg[0]:.3f}, Right: {nn_avg[1]:.3f}")
    print(f"Ground Truth - Left: {gt_avg[0]:.3f}, Right: {gt_avg[1]:.3f}")

    metrics = {
        'person_acc': person_acc,
        'group_acc': group_acc,
        'avg_person_class_acc': avg_person_class_acc,
        'avg_group_class_acc': avg_group_class_acc,
        'soft_consistency': soft_consistency,
        'group_aligned_consistency': group_aligned_consistency,
        'person_class_accs': person_class_accs,
        'group_class_accs': group_class_accs,
        'heuristic_avg_left': heuristic_avg[0],
        'heuristic_avg_right': heuristic_avg[1],
        'nn_avg_left': nn_avg[0],
        'nn_avg_right': nn_avg[1],
        'gt_avg_left': gt_avg[0],
        'gt_avg_right': gt_avg[1]
    }

    heuristic_winner_acc = np.mean(np.argmax(all_heuristic_probs, axis=1) == group_implied_winner) * 100
    nn_winner_acc = np.mean(np.argmax(all_rally_probs, axis=1) == group_implied_winner) * 100
    print(f"\nRally Winner Accuracy:")
    print(f"Heuristic: {heuristic_winner_acc:.2f}%")
    print(f"Neural Net: {nn_winner_acc:.2f}%")
    metrics.update({'heuristic_winner_acc': heuristic_winner_acc, 'nn_winner_acc': nn_winner_acc})

    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate volleyball action recognition model with rally prediction")
    parser.add_argument("--model_path", type=str, default="output/checkpoint_epoch_60_train_alexnet.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Path to dataset directory")
    parser.add_argument("--tolerance", type=float, default=0.25,
                        help="Tolerance for filtered soft consistency score")
    args = parser.parse_args()

    from dataset import VolleyballDataset, test_transform, TEST_VIDEOS

    data_dir = Path(args.data_dir)
    videos_dir = data_dir / "videos"
    annotations_dir = data_dir / "volleyball_tracking_annotation"

    test_dataset = VolleyballDataset(video_ids=TEST_VIDEOS, videos_dir=videos_dir, annotations_dir=annotations_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    metrics = evaluate_model(args.model_path, test_loader, tolerance=args.tolerance)
