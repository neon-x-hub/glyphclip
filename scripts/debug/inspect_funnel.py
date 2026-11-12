import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glyphclip.canvas.renderer import DeterministicCanvasRenderer
import numpy as np
import cv2

renderer = DeterministicCanvasRenderer()
font_path = r'C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\FunnelDisplay-VariableFont_wght.ttf'

test_chars = ['f', 'F', 'u', 'n', 'e', 'l', 'a', 'g', 'o']
for i in range(6):
    crop = cv2.imread(f'debug_crops/crop_{i}.png', cv2.IMREAD_GRAYSCALE)
    best_ch = None
    best_diff = 999
    for ch in test_chars:
        try:
            rendered = renderer.render_glyph_numpy(font_path, ch)
            diff = np.abs(crop.astype(float) - rendered.astype(float)).mean()
            if diff < best_diff:
                best_diff = diff
                best_ch = ch
        except Exception:
            pass
    print(f"Crop {i} best visual match: '{best_ch}' (diff: {best_diff:.1f})")
