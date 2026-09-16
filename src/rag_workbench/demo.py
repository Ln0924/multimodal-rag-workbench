from __future__ import annotations

import shutil
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


def project_root() -> Path:
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (candidate / "examples" / "atlas.md").is_file():
            return candidate
    raise RuntimeError("请在包含 examples 的项目目录执行演示")


def generate_demo(root: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("atlas.md", "retrieval.md"):
        shutil.copyfile(root / "examples" / name, output / name)
    native = output / "atlas-spec.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 70), "Atlas A1 - Original Demo Specification", fontsize=20)
        page.insert_text((60, 120), "Warranty: 24 months. Power: 65W. Input: USB-C.", fontsize=14)
        page.insert_text((60, 160), "Operating temperature: 0 to 40 degrees Celsius.", fontsize=14)
        pdf.save(native)
    image = Image.new("RGB", (1400, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=38)
    title_font = ImageFont.load_default(size=48)
    draw.text((55, 40), "Atlas A1 - Firmware Update", fill="black", font=title_font)
    steps = ["1. Connect USB-C", "2. Open settings", "3. Select firmware update"]
    for index, text in enumerate(steps):
        y = 145 + index * 145
        draw.rounded_rectangle((50, y, 1300, y + 95), radius=12, outline="black", width=3)
        draw.text((80, y + 20), text, fill="black", font=font)
    picture = output / "maintenance.png"
    image.save(picture)
    scan = output / "scanned-maintenance.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=700, height=325)
        page.insert_image(page.rect, filename=str(picture))
        pdf.save(scan)
    return [output / "atlas.md", output / "retrieval.md", native, picture, scan]
