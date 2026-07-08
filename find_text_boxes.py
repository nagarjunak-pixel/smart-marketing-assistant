from PIL import Image, ImageOps
import numpy as np
import os

img1_path = "/Users/venkataswaraswamy/.gemini/antigravity-ide/brain/68309493-fed5-4480-beae-c7a9b8bd9013/media__1782413787959.png"
img2_path = "/Users/venkataswaraswamy/.gemini/antigravity-ide/brain/68309493-fed5-4480-beae-c7a9b8bd9013/media__1782413814671.png"
out_dir = "/Users/venkataswaraswamy/.gemini/antigravity-ide/brain/68309493-fed5-4480-beae-c7a9b8bd9013"

def crop_tight_text(img_path, prefix):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
    img = Image.open(img_path)
    w, h = img.size
    
    # We know the text boxes are centered horizontally. Let's crop width from 25% to 75%
    left = int(w * 0.25)
    right = int(w * 0.75)
    
    # Let's analyze the vertical slice in the middle
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    
    # Calculate row-wise mean intensity in the horizontal range of the dialog
    # The dialog itself is dark in dark mode. The background of input boxes is even darker.
    # Text characters are white/light gray (light pixels in dark boxes).
    # Let's compute the standard deviation of each row in the center width.
    row_stds = [np.std(arr[y, left:right]) for y in range(h)]
    
    # Find peaks in standard deviation where text is present
    # We expect 2 main text regions corresponding to the two keys.
    # Let's search in the middle 50% of the image height (from 35% to 85% height)
    search_start = int(h * 0.35)
    search_end = int(h * 0.85)
    
    # Smooth the stds with a moving average of window 5 to reduce noise
    window = 5
    smoothed = np.convolve(row_stds, np.ones(window)/window, mode='same')
    
    # Let's print out rows where there's high standard deviation
    # We can split the search region into two halves to find the two fields
    mid = (search_start + search_end) // 2
    
    # Field 1 is in the top half
    f1_y = search_start + np.argmax(smoothed[search_start:mid])
    # Field 2 is in the bottom half
    f2_y = mid + np.argmax(smoothed[mid:search_end])
    
    print(f"{prefix}: Found text at row {f1_y} and row {f2_y}")
    
    # Crop a window of 35 pixels height around each detected row
    h_win = 22
    crop1 = img.crop((left, f1_y - h_win, right, f1_y + h_win))
    crop2 = img.crop((left, f2_y - h_win, right, f2_y + h_win))
    
    crop1.save(os.path.join(out_dir, f"{prefix}_tight1.png"))
    crop2.save(os.path.join(out_dir, f"{prefix}_tight2.png"))
    
    # Render ASCII
    render_ascii(crop1, f"{prefix} Field 1 (Tight)")
    render_ascii(crop2, f"{prefix} Field 2 (Tight)")

def render_ascii(img, title):
    img = ImageOps.grayscale(img)
    w, h = img.size
    # Resize for terminal display
    width = 110
    height = int(width * (h / w) * 0.5)
    img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
    pixels = img_resized.getdata()
    
    # Since text is light and box is dark:
    # Use dark characters for light pixels, and space for dark pixels
    chars = " .:-=+*#%@"
    num_chars = len(chars)
    
    ascii_str = []
    for i, p in enumerate(pixels):
        # Scale pixel value (0-255)
        val = p / 255.0
        idx = int(val * (num_chars - 1))
        ascii_str.append(chars[idx])
        if (i + 1) % width == 0:
            ascii_str.append("\n")
            
    print(f"\n--- {title} ---")
    print("".join(ascii_str))

crop_tight_text(img1_path, "api_keys")
crop_tight_text(img2_path, "access_tokens")
