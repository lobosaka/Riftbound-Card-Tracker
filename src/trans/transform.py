from PIL import Image
import os

card_dir = 'data/card_images'
card_list = os.listdir(card_dir)

def check_distinct_ratio_mode():

    '''
    Checks for all possible ratios and modes like
    RGB or RGBA
    '''

    distinct_ratio = set()
    distinct_mode = set()

    for card in card_list:
        file_path = os.path.join(card_dir, card)
        with Image.open(file_path) as img:
            width, height = img.size
            ratio = (width, height)
            distinct_ratio.add(ratio)
            mode = img.mode
            distinct_mode.add(mode)

    print(f"Distinct Ratios: {distinct_ratio}")
    print(f"Distinct Modes: {distinct_mode}")

def transform_pictures():

    '''
    Padding on the shorter side of the picture with
    black pixels and downsampling (resizing) to 224x224 pixels
    '''

    output_dir = "data/card_images_processed"
    os.makedirs(output_dir, exist_ok=True)

    i = 0

    for card in card_list:

        i += 1

        file_path = os.path.join(card_dir, card)
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            width, height = img_rgb.size
        max_side = max(width, height)
        # Create new symmetric picture with black pixels
        new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        # Calculate position of the picture from left and top to center
        left = (max_side - width) // 2
        top = (max_side - height) // 2
        # Put original picture on top of new picture with left and top margins as defined
        new_img.paste(img_rgb, (left, top))
        # Downsample new image to 224 x 224 pixels with LANCZOS algorithm
        final_img = new_img.resize((224, 224), Image.Resampling.LANCZOS)
        output_path = os.path.join(output_dir, card)
        final_img.save(output_path, optimize=True)

        if i % 10 == 0:
            print(f"{i} Karten transformiert!")


#check_distinct_ratio_mode()
transform_pictures()