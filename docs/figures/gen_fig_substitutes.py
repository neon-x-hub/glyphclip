"""Generate publication-quality Figure 6: Leave-One-Out Typographic Substitute Retrieval Grid.

Renders crisp, non-overlapping specimen cards for four distinct typographic genres:
  Row 1: Pixel & Bitmap Typography
  Row 2: Geometric Sans-Serif
  Row 3: Neo-Grotesque Sans-Serif
  Row 4: Transitional & Modern Serif

Each card features a dedicated metadata header (Rank, Font Name, and mIoU / CD badges)
and a completely unobstructed, centered rendering of 'Hamburgefonstiv' in the card body.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_FIGS = PROJECT_ROOT / "docs" / "figures"
REPORT_FIGS = PROJECT_ROOT / "report" / "figures"
DOCS_FIGS.mkdir(parents=True, exist_ok=True)
REPORT_FIGS.mkdir(parents=True, exist_ok=True)

# Colors
C_BG_PAGE = (255, 255, 255)
C_TEXT_MAIN = (15, 23, 42)      # slate-900
C_TEXT_MUTED = (71, 85, 105)    # slate-600
C_BORDER_DEFAULT = (226, 232, 240) # slate-200

# Rank styling
RANK_STYLES = {
    "query": {
        "border": (100, 116, 139),
        "header_bg": (241, 245, 249),
        "header_text": (15, 23, 42),
        "badge_bg": (226, 232, 240),
        "badge_text": (51, 65, 85),
    },
    "target": {
        "border": (220, 38, 38),
        "header_bg": (254, 242, 242),
        "header_text": (153, 27, 27),
        "badge_bg": (254, 202, 202),
        "badge_text": (185, 28, 28),
    },
    "rank1": {
        "border": (22, 163, 74),
        "header_bg": (240, 253, 244),
        "header_text": (20, 83, 45),
        "badge_bg": (220, 252, 231),
        "badge_text": (22, 101, 52),
    },
    "rank2": {
        "border": (37, 99, 235),
        "header_bg": (239, 246, 255),
        "header_text": (30, 58, 138),
        "badge_bg": (219, 234, 254),
        "badge_text": (30, 64, 175),
    },
    "rank3": {
        "border": (100, 116, 139),
        "header_bg": (248, 250, 252),
        "header_text": (30, 41, 59),
        "badge_bg": (226, 232, 240),
        "badge_text": (71, 85, 105),
    },
}


def get_ui_fonts():
    try:
        f_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 20)
        f_header = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 15)
        f_sub = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 12)
        f_badge = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 11)
        f_col = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 14)
    except Exception:
        f_title = f_header = f_sub = f_badge = f_col = ImageFont.load_default()
    return f_title, f_header, f_sub, f_badge, f_col


def render_query_card(img_input: Image.Image, genre_name: str, width: int = 340, height: int = 150) -> Image.Image:
    """Render the query image card."""
    _, f_header, f_sub, f_badge, _ = get_ui_fonts()
    card = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(card)

    style = RANK_STYLES["query"]

    # Header bar
    draw.rectangle([0, 0, width - 1, 38], fill=style["header_bg"])
    draw.rectangle([0, 0, width - 1, height - 1], outline=style["border"], width=1)
    draw.line([0, 38, width - 1, 38], fill=style["border"], width=1)

    draw.text((10, 10), "Visual Query Crop", font=f_header, fill=style["header_text"])

    # Badge on right
    badge_str = f"[{genre_name}]"
    bbox_b = draw.textbbox((0, 0), badge_str, font=f_badge)
    w_b = bbox_b[2] - bbox_b[0]
    draw.text((width - w_b - 10, 12), badge_str, font=f_badge, fill=style["badge_text"])

    # Fit image into body area [0, 39, width, height]
    body_w = width - 20
    body_h = height - 50
    aspect = img_input.width / max(1, img_input.height)

    if aspect > (body_w / body_h):
        new_w = body_w
        new_h = int(new_w / aspect)
    else:
        new_h = body_h
        new_w = int(new_h * aspect)

    resized = img_input.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x_pos = 10 + (body_w - new_w) // 2
    y_pos = 44 + (body_h - new_h) // 2

    # Draw subtle inner outline for crop
    draw.rectangle([x_pos - 1, y_pos - 1, x_pos + new_w, y_pos + new_h], outline=(203, 213, 225), width=1)
    card.paste(resized, (x_pos, y_pos))

    return card


def render_specimen_card(
    font_path: Path,
    font_name: str,
    rank_key: str,
    rank_title: str,
    badge_text: str,
    width: int = 490,
    height: int = 150,
) -> Image.Image:
    """Render a specimen card with separate header and completely clean template text."""
    _, f_header, f_sub, f_badge, _ = get_ui_fonts()
    card = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(card)

    style = RANK_STYLES[rank_key]

    # Header bar
    draw.rectangle([0, 0, width - 1, 38], fill=style["header_bg"])
    draw.rectangle([0, 0, width - 1, height - 1], outline=style["border"], width=2 if "rank1" in rank_key else 1)
    draw.line([0, 38, width - 1, 38], fill=style["border"], width=1)

    # Title: "Rank #1: Century Gothic"
    title_str = f"{rank_title}: {font_name}" if rank_title else font_name
    draw.text((12, 10), title_str, font=f_header, fill=style["header_text"])

    # Badge: e.g. "mIoU: 74.8% | CD: 2.18 px" or "[EXCLUDED FROM INDEX]"
    if badge_text:
        bbox_b = draw.textbbox((0, 0), badge_text, font=f_badge)
        w_b = bbox_b[2] - bbox_b[0]
        h_b = bbox_b[3] - bbox_b[1]
        bx = width - w_b - 12
        by = 8
        # Draw pill background
        draw.rounded_rectangle([bx - 6, by - 2, width - 6, by + h_b + 4], radius=3, fill=style["badge_bg"])
        draw.text((bx - 2, by), badge_text, font=f_badge, fill=style["badge_text"])

    # Specimen string
    text_spec = "Hamburgefonstiv"
    try:
        # Auto-size font to fit body perfectly
        spec_size = 36
        font = ImageFont.truetype(str(font_path), spec_size)
        bbox_t = draw.textbbox((0, 0), text_spec, font=font)
        w_t = bbox_t[2] - bbox_t[0]

        # If too wide, downscale slightly
        max_allowed_w = width - 24
        if w_t > max_allowed_w:
            spec_size = int(spec_size * (max_allowed_w / w_t))
            font = ImageFont.truetype(str(font_path), spec_size)
            bbox_t = draw.textbbox((0, 0), text_spec, font=font)
            w_t = bbox_t[2] - bbox_t[0]

        h_t = bbox_t[3] - bbox_t[1]
        body_avail_h = height - 42
        x_s = max(10, (width - w_t) // 2)
        y_s = 40 + max(4, (body_avail_h - h_t) // 2)
        draw.text((x_s, y_s), text_spec, font=font, fill=(15, 23, 42))
    except Exception as e:
        draw.text((20, 80), f"Specimen: [{font_name}]", font=f_sub, fill=(100, 116, 139))

    return card


def create_synthetic_query_crop(font_path: Path, word: str, size: int = 56) -> Image.Image:
    """Create realistic query crop with slight blur and padding."""
    img = Image.new("RGB", (len(word) * size // 2 + 70, size + 36), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(str(font_path), size=size)
        draw.text((20, 12), word, fill=(24, 28, 36), font=font)
    except Exception:
        pass
    arr = cv2.GaussianBlur(np.array(img), (3, 3), 0.4)
    return Image.fromarray(arr)


def build_figure_6():
    print("Generating refined publication Figure 6...")

    f_title, f_header, f_sub, f_badge, f_col = get_ui_fonts()

    # Define the 4 genre rows with strictly verified actual font files
    rows_def = [
        {
            "genre": "Pixel & Bitmap",
            "query_img": Image.open(PROJECT_ROOT / "ENSIA@PixelOperator8.png"),
            "target_name": "Pixel Operator 8",
            "target_path": Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\PixelOperator8.ttf"),
            "substitutes": [
                ("rank1", "Rank #1", "Silkscreen", Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\Silkscreen-Regular.ttf"), "mIoU: 50.6% | CD: 6.73 px"),
                ("rank2", "Rank #2", "Minecraft", Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\Minecraft.otf"), "mIoU: 77.4% | CD: 2.09 px"),
                ("rank3", "Rank #3", "Be Vietnam Pro [Suppressed]", Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\BeVietnamPro-Regular.ttf"), "mIoU: 53.4% | CD: 5.81 px"),
            ],
        },
        {
            "genre": "Geometric Sans",
            "query_img": Image.open(PROJECT_ROOT / "funnel.png"),
            "target_name": "Funnel Display",
            "target_path": Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\FunnelDisplay-VariableFont_wght.ttf"),
            "substitutes": [
                ("rank1", "Rank #1", "Century Gothic", Path(r"C:\Windows\Fonts\GOTHIC.TTF"), "mIoU: 56.5% | CD: 4.01 px"),
                ("rank2", "Rank #2", "Bahnschrift", Path(r"C:\Windows\Fonts\bahnschrift.ttf"), "mIoU: 53.0% | CD: 5.31 px"),
                ("rank3", "Rank #3", "Segoe UI", Path(r"C:\Windows\Fonts\segoeui.ttf"), "mIoU: 62.1% | CD: 3.66 px"),
            ],
        },
        {
            "genre": "Neo-Grotesque",
            "query_img": create_synthetic_query_crop(Path(r"C:\Windows\Fonts\arial.ttf"), "Helvetica"),
            "target_name": "Arial",
            "target_path": Path(r"C:\Windows\Fonts\arial.ttf"),
            "substitutes": [
                ("rank1", "Rank #1", "Segoe UI", Path(r"C:\Windows\Fonts\segoeui.ttf"), "mIoU: 72.0% | CD: 2.75 px"),
                ("rank2", "Rank #2", "Calibri", Path(r"C:\Windows\Fonts\calibri.ttf"), "mIoU: 72.1% | CD: 2.85 px"),
                ("rank3", "Rank #3", "Verdana", Path(r"C:\Windows\Fonts\verdana.ttf"), "mIoU: 77.1% | CD: 2.24 px"),
            ],
        },
        {
            "genre": "Transitional Serif",
            "query_img": create_synthetic_query_crop(Path(r"C:\Windows\Fonts\times.ttf"), "Typography"),
            "target_name": "Times New Roman",
            "target_path": Path(r"C:\Windows\Fonts\times.ttf"),
            "substitutes": [
                ("rank1", "Rank #1", "Georgia", Path(r"C:\Windows\Fonts\georgia.ttf"), "mIoU: 75.2% | CD: 1.92 px"),
                ("rank2", "Rank #2", "Libre Baskerville", Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\LibreBaskerville-Regular.ttf"), "mIoU: 72.5% | CD: 2.26 px"),
                ("rank3", "Rank #3", "Cambria", Path(r"C:\Windows\Fonts\cambriab.ttf"), "mIoU: 58.7% | CD: 4.84 px"),
            ],
        },
    ]

    # Canvas geometry (header title cropped so LaTeX \caption alone provides the caption)
    w_query = 340
    w_card = 490
    card_h = 136
    gap_x = 16
    gap_y = 18
    pad_left = 32
    pad_top = 48
    pad_bottom = 20

    total_w = pad_left * 2 + w_query + 4 * w_card + 4 * gap_x
    total_h = pad_top + 4 * card_h + 3 * gap_y + pad_bottom

    page = Image.new("RGB", (total_w, total_h), color=C_BG_PAGE)
    draw = ImageDraw.Draw(page)

    # Column Headers
    col_x_starts = [
        pad_left,
        pad_left + w_query + gap_x,
        pad_left + w_query + gap_x + (w_card + gap_x),
        pad_left + w_query + gap_x + 2 * (w_card + gap_x),
        pad_left + w_query + gap_x + 3 * (w_card + gap_x),
    ]

    col_headers = [
        ("Visual Query Crop", "(Raster Input)"),
        ("Ground Truth Target", "(Excluded from Index)"),
        ("Rank #1 Recommended", "(Top Substitute)"),
        ("Rank #2 Recommended", "(Alternative Substitute)"),
        ("Rank #3 Recommended", "(Alternative / Contrast)"),
    ]

    for (h_top, h_bot), x in zip(col_headers, col_x_starts):
        draw.text((x, 16), h_top, font=f_col, fill=C_TEXT_MAIN)
        bbox_top = draw.textbbox((x, 16), h_top, font=f_col)
        draw.text((bbox_top[2] + 8, 18), h_bot, font=f_badge, fill=C_TEXT_MUTED)

    # Render Rows
    for r_idx, r_data in enumerate(rows_def):
        y_row = pad_top + r_idx * (card_h + gap_y)

        # 1. Query Card
        q_card = render_query_card(r_data["query_img"], r_data["genre"], width=w_query, height=card_h)
        page.paste(q_card, (col_x_starts[0], y_row))

        # 2. Target Card (Excluded)
        t_card = render_specimen_card(
            r_data["target_path"],
            r_data["target_name"],
            rank_key="target",
            rank_title="Target",
            badge_text="[EXCLUDED FROM INDEX]",
            width=w_card,
            height=card_h,
        )
        page.paste(t_card, (col_x_starts[1], y_row))

        # 3, 4, 5. Substitutes
        for s_idx, (r_key, r_title, f_name, f_path, badge_txt) in enumerate(r_data["substitutes"]):
            s_card = render_specimen_card(
                f_path,
                f_name,
                rank_key=r_key,
                rank_title=r_title,
                badge_text=badge_txt,
                width=w_card,
                height=card_h,
            )
            page.paste(s_card, (col_x_starts[s_idx + 2], y_row))

    # Save to high-res PNG and PDF
    for out_p in [
        DOCS_FIGS / "fig6_visual_substitutes.png",
        REPORT_FIGS / "fig6_visual_substitutes.png",
    ]:
        page.save(out_p, dpi=(300, 300))
        print(f"Saved sharp PNG to: {out_p}")

    # For PDF: save directly via PIL with 300 DPI
    for out_pdf in [
        DOCS_FIGS / "fig6_visual_substitutes.pdf",
        REPORT_FIGS / "fig6_visual_substitutes.pdf",
    ]:
        page.save(out_pdf, "PDF", resolution=300.0)
        print(f"Saved clean PDF to: {out_pdf}")


if __name__ == "__main__":
    build_figure_6()
