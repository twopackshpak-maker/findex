"""Create the 2–3 minute, captioned Lab 8 demonstration video."""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 8
BG, PANEL, LINE, WHITE, MUTED, BLUE, GREEN = (
    "#0b1220",
    "#111c2e",
    "#26374d",
    "#f1f5f9",
    "#a7b4c5",
    "#38bdf8",
    "#34d399",
)
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def card(title: str, subtitle: str, bullets: list[str], number: str) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((70, 44), "FINDEX  /  LAB 08", font=f(18, True), fill=BLUE)
    d.text((70, 102), title, font=f(42, True), fill=WHITE)
    d.text((72, 167), subtitle, font=f(23), fill=MUTED)
    d.rounded_rectangle((64, 220, 1216, 628), radius=22, fill=PANEL, outline=LINE, width=2)
    y = 266
    for item in bullets:
        d.ellipse((100, y + 8, 116, y + 24), fill=GREEN)
        d.text((140, y), item, font=f(23), fill=WHITE)
        y += 68
    d.text((72, 669), "Search · measure · optimize · explain", font=f(16), fill=MUTED)
    d.text((1130, 669), number, font=f(16, True), fill=BLUE)
    return im


def main() -> None:
    gif = Image.open("assets/search_demo.gif")
    gif_frames = []
    try:
        while True:
            gif_frames.append(gif.convert("RGB").copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    slides: list[tuple[Image.Image, int]] = [
        (
            card(
                "Findex, швидкий пошук у корпусі",
                "Профілювання · NumPy · memory mapping",
                [
                    "1200 документів · корпус 64.17 MiB",
                    "API p95: 840 ms при 20 одночасних клієнтах",
                    "Найбільше зафіксоване прискорення: 2.25×",
                ],
                "00:00",
            ),
            18,
        ),
        (
            card(
                "Живе ранжування",
                "Текстовий запит → релевантні документи",
                [
                    "Пошук `python concurrency inverted index`",
                    "Перші документи відповідають запиту за BM25",
                    "Порядок перевірено проти скалярного scorer-а з Лаби 3",
                ],
                "00:18",
            ),
            24,
        ),
        (
            card(
                "Що показав профайлер",
                "cProfile на індексації та пошуку",
                [
                    "Основне навантаження індексації — токенізація і Python-підрахунок tf",
                    "Regex та Unicode normalization частково працюють у native-коді",
                    "Короткий запит витрачає час на розбір і підготовку результатів",
                ],
                "00:42",
            ),
            22,
        ),
        (
            card(
                "NumPy там, де є робота",
                "int32 postings · ufunc · broadcasting · argpartition",
                [
            "Частий терм: 60.44 → 46.54 ms · 1.30×",
            "Рідкісний терм: 1.98 → 2.02 ms · прискорення майже немає",
            "Три терми: 0.013 → 0.028 ms · менший список не окупив виклики",
                ],
                "01:04",
            ),
            24,
        ),
        (
            card(
                "10× індекс у mmap",
            "12 000 документів · 419 990 posting entries",
                [
                    "Мапа зберігає doc_id, tf і довжини як int32",
                    "Три масиви займають 3.25 MiB на диску",
            "Медіанний пошук 0.066 ms; сторінки ОС уникають повної копії у Python heap",
                ],
                "01:28",
            ),
            22,
        ),
        (
            card(
                "Перевірка HTTP API",
                "500 запитів · concurrency=20 · 0 помилок",
                [
                    "19.87 s загалом · 25.2 запити/с",
                    "p50 776 ms · p95 840 ms · p99 1833 ms",
                    "На піку черга запитів збільшує затримку більше, ніж scorer",
                ],
                "01:50",
            ),
            20,
        ),
        (
            card(
                "Висновок",
                "Оптимізація має сенс там, де профіль показує обсяг",
                [
                    "Векторизація виграє на довгих posting lists",
                    "Для коротких списків Python overhead лишається помітним",
                    "Наступний крок: профіль на production hardware і контроль p95",
                ],
                "02:10",
            ),
            22,
        ),
    ]

    writer = imageio_ffmpeg.write_frames(
        "assets/lab8_demo.mp4",
        (W, H),
        fps=FPS,
        codec="libx264",
        quality=7,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=16,
    )
    writer.send(None)
    index = 0
    for slide_no, (base, seconds) in enumerate(slides):
        count = seconds * FPS
        for tick in range(count):
            frame = base.copy()
            if slide_no == 1:
                demo = gif_frames[(tick // 2) % len(gif_frames)].resize((800, 520))
                frame.paste(demo, (240, 190))
            writer.send(frame.tobytes())
            index += 1
    writer.close()
    print(
        f"Wrote assets/lab8_demo.mp4: {index / FPS:.1f} sec, "
        f"{Path('assets/lab8_demo.mp4').stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
