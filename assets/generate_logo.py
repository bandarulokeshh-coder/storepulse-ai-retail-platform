"""Generate assets/logo.png for StorePulse AI.

Run once:  python assets/generate_logo.py
Requires Pillow (installed automatically with Streamlit).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path(__file__).resolve().parent / "logo.png"
SIZE = 512
PRIMARY = (37, 99, 235, 255)  # #2563EB
DARK = (15, 23, 42, 255)  # #0F172A
WHITE = (255, 255, 255, 255)
GREEN = (16, 185, 129, 255)  # #10B981


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Rounded background square.
    draw.rounded_rectangle([8, 8, SIZE - 8, SIZE - 8], radius=96, fill=PRIMARY)

    # Vertical accent overlay (subtle two-tone look).
    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle([8, 8, SIZE - 8, SIZE - 8], radius=96, fill=(15, 23, 42, 90))
    overlay_draw.rounded_rectangle([8, SIZE // 2, SIZE - 8, SIZE - 8], radius=96, fill=(0, 0, 0, 0))
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    # Pulse line (the "pulse" in StorePulse).
    pulse_y = SIZE // 2 + 90
    points = [
        (56, pulse_y),
        (150, pulse_y),
        (185, pulse_y - 70),
        (225, pulse_y + 85),
        (262, pulse_y - 40),
        (300, pulse_y),
        (456, pulse_y),
    ]
    draw.line(points, fill=GREEN, width=16, joint="curve")

    # "SP" monogram.
    font = _font(200)
    text = "SP"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(((SIZE - text_w) / 2 - bbox[0], (SIZE // 2 - 40 - text_h) - bbox[1]), text, font=font, fill=WHITE)

    image.save(OUTPUT, "PNG")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()