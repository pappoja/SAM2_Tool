import os
from PIL import Image
import matplotlib.pyplot as plt
import ipywidgets as widgets

def annotate(frame_idx, video_dir, frame_names, video_segments):
    global data
    data = []

    image_path = os.path.join(video_dir, frame_names[frame_idx])
    img = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"Select Points and Assign Labels: Frame {frame_idx}")
    im = ax.imshow(img)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)

    if frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            if out_mask is not None and np.any(out_mask):
                show_mask(out_mask, ax, obj_id=out_obj_id)

    global current_points, current_labels, current_obj_id, current_label
    current_points = []
    current_labels = []
    current_obj_id = 1
    current_label = 1

    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            current_points.append([x, y])
            current_labels.append(current_label)
            ax.text(x, y, str(current_obj_id), color='blue' if current_label == 1 else 'red', fontsize=12, ha='center')
            fig.canvas.draw()
            existing_obj = next((item for item in data if item[0] == current_obj_id), None)
            if existing_obj:
                existing_obj[1].append([x, y])
                existing_obj[2].append(current_label)
            else:
                data.append((current_obj_id, [[x, y]], [current_label]))

    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    def update_label(change):
        global current_label
        current_label = int(change['new'])

    label_dropdown = widgets.Dropdown(
        options=[1, 0],
        value=current_label,
        description='Label:',
    )

    label_dropdown.observe(update_label, names='value')
    display(label_dropdown)

    def update_obj_id(change):
        global current_obj_id
        save_current_object_data()
        current_obj_id = int(change['new'])

    def save_current_object_data():
        global current_points, current_labels, data, current_obj_id
        if current_points and current_labels:
            existing_obj = next((item for item in data if item[0] == current_obj_id), None)
            if existing_obj:
                existing_obj[1].extend(current_points)
                existing_obj[2].extend(current_labels)
            else:
                data.append((current_obj_id, current_points.copy(), current_labels.copy()))
            current_points.clear()
            current_labels.clear()

    object_id_dropdown = widgets.Dropdown(
        options=[i for i in range(1, 51)],
        value=current_obj_id,
        description='Object ID:',
    )

    object_id_dropdown.observe(update_obj_id, names='value')
    display(object_id_dropdown)

    plt.show()

def make_prompts(data):
    prompts = {}
    for obj_id, points, labels in data:
        prompts[obj_id] = (
            np.array(points, dtype=np.float32),
            np.array(labels, np.int32)
        )
    return prompts

def add_prompts(prompts, frame_idx, predictor, inference_state, is_refinement=False):
    if is_refinement:
        predictor.reset_state(inference_state)
    for obj_id, (points, labels) in prompts.items():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels
        )
    return _, out_obj_ids, out_mask_logits
