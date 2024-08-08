import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'sam2'))

from sam2_annotation.video_loader import load_video
from sam2_annotation.visualization import show_mask, show_points, mask_to_bb
from sam2_annotation.annotation import annotate, make_prompts, add_prompts
from sam2_annotation.inference import propagate_masks, view_labeled_frames
from sam2.build_sam import build_sam2_video_predictor

model_size = "tiny"
sam2_checkpoint = f"sam2_hiera_{model_size}.pt"
model_cfg = f"sam2_hiera_{model_size[0]}.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
video_dir = 'data/frames'
input_video = 'data/input_video/marshawn.mp4'

load_video(input_video, video_dir, 1)

frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)

frame_idx = 0
annotate(frame_idx, video_dir, frame_names, {})

prompts = make_prompts(data)
_, out_obj_ids, out_mask_logits = add_prompts(prompts, frame_idx, predictor, inference_state)

plt.figure(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
    show_mask(mask, plt.gca(), obj_id=out_obj_id)

video_segments = propagate_masks(inference_state, predictor, start_frame_idx=15)

view_labeled_frames(15, frame_names, video_segments, video_dir)

frame_idx = 210
annotate(frame_idx, video_dir, frame_names, video_segments)
prompts = make_prompts(data)

fig_before, ax_before = plt.subplots(figsize=(12, 8))
ax_before.set_title(f"Frame {frame_idx} -- Before Refinement")
ax_before.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
for out_obj_id, out_mask in video_segments[frame_idx].items():
    if out_mask is not None and np.any(out_mask):
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
display(fig_before)
plt.close(fig_before)

_, out_obj_ids, out_mask_logits = add_prompts(prompts, frame_idx, predictor, inference_state, is_refinement=True)

fig_after, ax_after = plt.subplots(figsize=(12, 8))
ax_after.set_title(f"Frame {frame_idx} -- After Refinement")
ax_after.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
    show_mask(mask, plt.gca(), obj_id=out_obj_id)
display(fig_after)
plt.close(fig_after)

video_segments = propagate_masks(inference_state, predictor, video_segments, start_frame_idx=frame_idx)

view_labeled_frames(15, frame_names, video_segments, video_dir)

print("Annotation and tracking completed successfully!")
