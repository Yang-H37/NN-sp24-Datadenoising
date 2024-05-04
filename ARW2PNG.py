import rawpy
import imageio
import os
import glob

def convert_arw_to_png(arw_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List ARW files
    arw_files = glob.glob(os.path.join(arw_path, '*.ARW'))

    # Convert each ARW file to PNG
    for arw_file in arw_files:
        with rawpy.imread(arw_file) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            output_file = os.path.splitext(os.path.basename(arw_file))[0] + '.png'
            imageio.imsave(os.path.join(output_dir, output_file), rgb)

if __name__ == "__main__":
    arw_path = 'dataset/long/'
    output_dir = 'dataset/long_png/'
    convert_arw_to_png(arw_path, output_dir)
