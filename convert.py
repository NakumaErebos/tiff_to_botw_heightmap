import sys
import numpy as np
import tifffile
from PIL import Image

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

def normalize_tiff_log_exp(
    input_path, 
    output_path,         
    steps=70,           # Count of different heights
    log_strength=20.0,  # Magnitude of logarithmic height quantization
    show_outlines=True, # Activates Outlining samge heights
    hex_low="403205",   # The lowest Point will have this color
    hex_high="C7C9B3",  # The highest Point will have this Color
    border_low="322400",# The lowest Altitude will be outlined with tihs color
    border_high="626550"# The highest Point will be outlined with this color
):
    try:
        data = tifffile.imread(input_path).astype(np.float32)
    except Exception as e:
        print(f"Error on reading in the tiff-file: {e}")
        return

    if data.ndim > 2:
        data = data[:, :, 0]

    min_val = np.min(data)
    max_val = np.max(data)

    print(f"Highest Point ({hex_low}): {min_val}")
    print(f"Lowest Punkt ({hex_high}): {max_val}")
    print(f"Using {steps} Steps (Log-Heights x Exp-Colors, k={log_strength}). Outlines: {'On' if show_outlines else 'Of'}")

    rgb_low = hex_to_rgb(hex_low)
    rgb_high = hex_to_rgb(hex_high)
    b_low = hex_to_rgb(border_low)
    b_high = hex_to_rgb(border_high)

    if max_val == min_val:
        stepped = np.zeros_like(data, dtype=np.float32)
        color_factor = np.zeros_like(data, dtype=np.float32)
    else:
        normalized = (data - min_val) / (max_val - min_val)
        
        if log_strength > 0:
            log_normalized = np.log1p(normalized * log_strength) / np.log1p(log_strength)
        else:
            log_normalized = normalized

        stepped = np.floor(log_normalized * steps) / (steps - 1)
        stepped = np.clip(stepped, 0.0, 1.0)

        if log_strength > 0:
            color_factor = np.expm1(stepped * np.log1p(log_strength)) / log_strength
        else:
            color_factor = stepped

    norm_3d = np.expand_dims(color_factor, axis=-1)
    fill_rgb = rgb_low + norm_3d * (rgb_high - rgb_low)

    if show_outlines:
        border_rgb = b_low + norm_3d * (b_high - b_low)

        diff_right = np.zeros_like(stepped, dtype=bool)
        diff_down = np.zeros_like(stepped, dtype=bool)

        diff_right[:, :-1] = stepped[:, :-1] != stepped[:, 1:]
        diff_down[:-1, :] = stepped[:-1, :] != stepped[1:, :]

        is_border = diff_right | diff_down
        is_border_3d = np.expand_dims(is_border, axis=-1)
        
        final_rgb = np.where(is_border_3d, border_rgb, fill_rgb).astype(np.uint8)
    else:
        final_rgb = fill_rgb.astype(np.uint8)

    # 6. Speichern
    res_img = Image.fromarray(final_rgb, mode='RGB')
    res_img.save(output_path)
    print(f"Finished, Picture saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("How to use: python convert.py <input.tiff> <output.png> [steps] [log_strenght] [outlines_true_false]")
    else:
        steps_param = int(sys.argv[3]) if len(sys.argv) > 3 else 70
        log_param = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
        
        # Outlines steuern (Standard: true)
        outlines_param = True
        if len(sys.argv) > 5:
            outlines_param = sys.argv[5].lower() in ['true', '1', 'ja', 'yes']

        normalize_tiff_log_exp(
            sys.argv[1], 
            sys.argv[2], 
            steps=steps_param, 
            log_strength=log_param, 
            show_outlines=outlines_param
        )