from PIL import Image, ImageDraw
import random
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

def generate_static_image(width, height, circle_radius, circle_center_x, circle_center_y, output_filepath, current_brightness_shift):
    """
    Generates a grayscale image with 'snow' static.
    The background static is biased towards darker shades (blacks and dark greys).
    The circular region has static that is shifted by `current_brightness_shift`
    making its overall brightness vary subtly.

    Args:
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.
        circle_radius (int): The radius of the circular region.
        circle_center_x (int): The x-coordinate of the circle's center.
        circle_center_y (int): The y-coordinate of the circle's center.
        output_filepath (str): The full path including filename for the output image.
        current_brightness_shift (int): The specific brightness shift to apply to the circle pixels.
                                        Can be positive (lighter) or negative (darker).
    """

    img = Image.new('L', (width, height))
    pixels = img.load()
    max_background_grey = 150

    for y in range(height):
        for x in range(width):
            distance = ((x - circle_center_x)**2 + (y - circle_center_y)**2)**0.5
            base_static_value = random.randint(0, max_background_grey)

            if distance <= circle_radius:
                pixel_value = min(max(base_static_value + current_brightness_shift, 0), 255)
            else:
                pixel_value = base_static_value

            pixels[x, y] = pixel_value

    img.save(output_filepath)
    return img

def generate_single_sample(
    sample_index,
    img_width,
    img_height,
    circle_rad,
    images_dir,
    labels_dir
):
    """
    Generates a single image and its corresponding YOLO label file.
    This function is designed to be run by a process.
    """
    random_circle_center_x = random.randint(circle_rad, img_width - circle_rad)
    circle_center_y_coord = img_height // 2
    random_brightness_shift = 20 + random.randint(-4, 3)

    img_filename = f"image_{sample_index:05d}.png"
    label_filename = f"image_{sample_index:05d}.txt"

    img_filepath = os.path.join(images_dir, img_filename)
    label_filepath = os.path.join(labels_dir, label_filename)

    generate_static_image(
        width=img_width,
        height=img_height,
        circle_radius=circle_rad,
        circle_center_x=random_circle_center_x,
        circle_center_y=circle_center_y_coord,
        output_filepath=img_filepath,
        current_brightness_shift=random_brightness_shift
    )

    # Generate the YOLO label file
    class_id = 0
    normalized_center_x = random_circle_center_x / img_width
    normalized_center_y = circle_center_y_coord / img_height
    normalized_width = (2 * circle_rad) / img_width
    normalized_height = (2 * circle_rad) / img_height

    with open(label_filepath, 'w') as f:
        f.write(f"{class_id} {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}\n")

    return True

def generate_yolo_dataset_multiprocessed(
    base_output_dir="yolo_captcha_data",
    img_width=400,
    img_height=200,
    circle_rad=60,
    num_train_samples=8000,
    num_val_samples=1000,
    num_test_samples=1000,
    max_workers=os.cpu_count()
):
    """
    Generates a YOLO-formatted dataset
    """
    splits = {
        "train": num_train_samples,
        "val": num_val_samples,
        "test": num_test_samples
    }

    print(f"Starting YOLO dataset generation in '{base_output_dir}' using {max_workers} processes...")

    for split_name, num_samples in splits.items():
        split_base_dir = os.path.join(base_output_dir, split_name)
        images_dir = os.path.join(split_base_dir, "images")
        labels_dir = os.path.join(split_base_dir, "labels")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        print(f"Created directories for '{split_name}': {images_dir} and {labels_dir}")

        print(f"Generating {num_samples} samples for '{split_name}' split...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_single_sample,
                                       i, img_width, img_height, circle_rad,
                                       images_dir, labels_dir)
                       for i in range(num_samples)]

            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                    if (i + 1) % (num_samples // 10 or 1) == 0:
                        print(f"  Generated {i + 1}/{num_samples} for {split_name}")
                        sys.stdout.flush()
                except Exception as exc:
                    print(f"Generated sample threw an exception: {exc}")
                    sys.stdout.flush()

    print("\nDataset generation complete!")
    print(f"Dataset structure created in: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    preview_img_width = 400
    preview_img_height = 200
    preview_circle_rad = 60
    preview_circle_center_x_coord = random.randint(preview_circle_rad, preview_img_width - preview_circle_rad)
    preview_circle_center_y_coord = preview_img_height // 2 # Y-center fixed for preview
    preview_output_filename = "captcha_static_preview.png"
    preview_label_filename = "captcha_static_preview.txt"

    preview_brightness_shift = 20 + random.randint(-4, 3)

    img_to_draw_on = generate_static_image(
        width=preview_img_width,
        height=preview_img_height,
        circle_radius=preview_circle_rad,
        circle_center_x=preview_circle_center_x_coord,
        circle_center_y=preview_circle_center_y_coord,
        output_filepath=preview_output_filename,
        current_brightness_shift=preview_brightness_shift
    )

    class_id = 0
    normalized_center_x = preview_circle_center_x_coord / preview_img_width
    normalized_center_y = preview_circle_center_y_coord / preview_img_height
    normalized_width = (2 * preview_circle_rad) / preview_img_width
    normalized_height = (2 * preview_circle_rad) / preview_img_height

    with open(preview_label_filename, 'w') as f:
        f.write(f"{class_id} {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}\n")

    draw = ImageDraw.Draw(img_to_draw_on)

    x_center_px = int(normalized_center_x * preview_img_width)
    y_center_px = int(normalized_center_y * preview_img_height)
    bbox_width_px = int(normalized_width * preview_img_width)
    bbox_height_px = int(normalized_height * preview_img_height)

    x1 = x_center_px - (bbox_width_px // 2)
    y1 = y_center_px - (bbox_height_px // 2)
    x2 = x_center_px + (bbox_width_px // 2)
    y2 = y_center_px + (bbox_height_px // 2)

    draw.rectangle([x1, y1, x2, y2], outline=255, width=2)

    img_to_draw_on.save(preview_output_filename)
    print(f"Preview image '{preview_output_filename}' with bounding box generated.")


    try:
        os.startfile(preview_output_filename)
    except AttributeError:
        try:
            subprocess.run(['open', preview_output_filename]) # For macOS
        except FileNotFoundError:
            try:
                subprocess.run(['xdg-open', preview_output_filename]) # For Linux
            except FileNotFoundError:
                print("Could not open preview image automatically. Please check 'captcha_static_preview.png' manually.")
    except Exception as e:
        print(f"An error occurred while trying to open the preview image: {e}")
        print("Please open 'captcha_static_preview.png' manually to view it.")

    print("\n--- Starting YOLO Dataset Generation ---")
    generate_yolo_dataset_multiprocessed(
        base_output_dir="yolo_captcha_data",
        img_width=400,
        img_height=200,
        circle_rad=60,
        num_train_samples=8000,
        num_val_samples=1000,
        num_test_samples=1000,
        max_workers=os.cpu_count()
    )
