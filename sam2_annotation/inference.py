def propagate_masks(inference_state, predictor, video_segments={}, start_frame_idx=0):
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments

def view_labeled_frames(frame_stride, frame_names, video_segments, video_dir, cols=4):
    import matplotlib.pyplot as plt
    import math
    from PIL import Image
    plt.close("all")
    frames_to_display = list(range(0, len(frame_names), frame_stride))
    rows = math.ceil(len(frames_to_display) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes]

    for ax, out_frame_idx in zip(axes, frames_to_display):
        ax.set_title(f"frame {out_frame_idx}")
        ax.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                if out_mask is not None and np.any(out_mask):
                    show_mask(out_mask, ax, obj_id=out_obj_id)
        ax.axis('off')

    for ax in axes[len(frames_to_display):]:
        ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.tight_layout(pad=0.5)
    display(fig)
    plt.close(fig)
