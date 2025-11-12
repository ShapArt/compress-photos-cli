#!/usr/bin/env python3
"""
compress_photos.py — рекурсивно сжимает фото в папке.

НОВОЕ:
- Исправлен баг: при дефолтном subsampling="keep" Pillow падал, если исходник не настоящий JPEG.
  Теперь по умолчанию --subsampling auto: для JPEG сохраняем исходный субсемплинг, для остальных — используем 1 (4:2:0).
- Добавлен автоматический фолбэк: если при сохранении JPEG с subsampling произошла ошибка, повторяем с subsampling=1.

Остальное:
- Рекурсивный обход подкаталогов.
- Вывод в отдельную папку <source>_compressed (структура зеркалится). Оригиналы НЕ трогаем.
- JPEG/WebP/PNG/TIFF/HEIC/HEIF/BMP: адекватные параметры сжатия.
- Сохраняем EXIF/ICC по возможности.
- HEIC/HEIF автоматически конвертируются в JPEG (при наличии pillow-heif).
- Можно ограничить длинную сторону (--max-size), например 2560 px.
- Если файл после сжатия больше исходного — оставляем оригинал (копируем как есть).
- Параллельная обработка.
Зависимости: pillow (обязательно), pillow-heif (для HEIC/HEIF, опционально).

Установка:
  pip install --upgrade pillow pillow-heif
    python compress_photos.py "D:/Photos" --max-size 2560 --quality 84 --workers 8
  python compress_photos.py ./input --out ./output_min --convert-to-jpeg --max-size 3000
  python compress_photos.py ./input --overwrite --suffix ""  # перезаписать рядом без суффикса
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image, ImageOps

# Попытка подключить HEIC/HEIF поддержку (опционально)
_HEIF_AVAILABLE = False
try:
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    _HEIF_AVAILABLE = True
except Exception:
    _HEIF_AVAILABLE = False


PHOTO_EXTS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
    ".heic",
    ".heif",
}


def iter_image_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in PHOTO_EXTS:
            yield p


def has_alpha(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    if img.mode == "P":
        return "transparency" in img.info
    return False


def flatten_alpha(img: Image.Image, bg_rgb: Tuple[int, int, int]) -> Image.Image:
    if not has_alpha(img):
        return img.convert("RGB") if img.mode != "RGB" else img
    bg = Image.new("RGB", img.size, bg_rgb)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg.paste(img, mask=img.split()[-1])
    return bg


def _save_with_retry(
    img: Image.Image, path: Path, save_kwargs: dict, fallback_for_keep: bool
) -> None:
    try:
        img.save(path, **save_kwargs)
    except Exception as e:
        msg = str(e).lower()
        if fallback_for_keep and ("keep" in msg or "subsampling" in msg):
            # Подменяем на универсальный subsampling=1 и пробуем ещё раз
            save_kwargs = dict(save_kwargs)
            save_kwargs["subsampling"] = 1
            img.save(path, **save_kwargs)
        else:
            raise


def safe_save(
    img: Image.Image,
    out_path: Path,
    original_path: Path,
    fmt: str,
    keep_exif: bool,
    quality: int,
    subsampling: Optional[Union[int, str]],
    png_level: int,
    skip_if_larger: bool,
) -> Tuple[bool, int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    exif = img.info.get("exif", b"") if keep_exif and "exif" in img.info else None
    icc = img.info.get("icc_profile") if "icc_profile" in img.info else None

    save_kwargs = {}
    fmt_upper = fmt.upper()

    if fmt_upper in ("JPG", "JPEG"):
        save_kwargs.update(
            {
                "format": "JPEG",
                "quality": int(quality),
                "optimize": True,
                "progressive": True,
            }
        )
        if subsampling is not None:
            save_kwargs["subsampling"] = subsampling
        if exif:
            save_kwargs["exif"] = exif
        if icc:
            save_kwargs["icc_profile"] = icc

        _save_with_retry(img, tmp_path, save_kwargs, fallback_for_keep=True)

    elif fmt_upper == "PNG":
        save_kwargs.update(
            {
                "format": "PNG",
                "optimize": True,
                "compress_level": int(png_level),
            }
        )
        if icc:
            save_kwargs["icc_profile"] = icc
        img.save(tmp_path, **save_kwargs)

    elif fmt_upper == "WEBP":
        save_kwargs.update(
            {
                "format": "WEBP",
                "quality": int(quality),
                "method": 6,
            }
        )
        if exif:
            save_kwargs["exif"] = exif
        img.save(tmp_path, **save_kwargs)

    elif fmt_upper in ("TIFF", "TIF"):
        save_kwargs.update(
            {
                "format": "TIFF",
                "compression": "tiff_adobe_deflate",
                "dpi": img.info.get("dpi", (72, 72)),
            }
        )
        if icc:
            save_kwargs["icc_profile"] = icc
        img.save(tmp_path, **save_kwargs)

    else:
        # fallback: сохраняем как JPEG
        save_kwargs.update(
            {
                "format": "JPEG",
                "quality": int(quality),
                "optimize": True,
                "progressive": True,
            }
        )
        if subsampling is not None:
            save_kwargs["subsampling"] = subsampling
        if exif:
            save_kwargs["exif"] = exif
        if icc:
            save_kwargs["icc_profile"] = icc

        _save_with_retry(img, tmp_path, save_kwargs, fallback_for_keep=True)

    new_size = tmp_path.stat().st_size
    orig_size = original_path.stat().st_size

    if not skip_if_larger or new_size <= orig_size:
        tmp_path.replace(out_path)
        return True, new_size, orig_size

    tmp_path.unlink(missing_ok=True)
    if out_path.resolve() != original_path.resolve():
        shutil.copy2(original_path, out_path)
    return False, orig_size, orig_size


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def parse_bg(bg: str) -> Tuple[int, int, int]:
    bg = bg.strip().lstrip("#")
    if len(bg) == 3:
        bg = "".join(ch * 2 for ch in bg)
    if len(bg) != 6:
        raise argparse.ArgumentTypeError("bg color must be like '#ffffff' or 'fff'")
    r = int(bg[0:2], 16)
    g = int(bg[2:4], 16)
    b = int(bg[4:6], 16)
    return (r, g, b)


def parse_subsampling(val: str) -> Union[int, str]:
    v = str(val).strip().lower()
    if v in ("auto", "keep"):
        return v
    if v in ("0", "1", "2"):
        return int(v)
    raise argparse.ArgumentTypeError("subsampling must be one of: auto, keep, 0, 1, 2")


def main():
    parser = argparse.ArgumentParser(description="Сжать фото рекурсивно в папке.")
    parser.add_argument("src", type=Path, help="Исходная папка с фото")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Куда класть результат (по умолчанию <src>_compressed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать файлы на месте (Осторожно!)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_min",
        help="Суффикс для имён (только если не --overwrite)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2560,
        help="Максимальный размер длинной стороны, px (0 = не менять)",
    )
    parser.add_argument(
        "--quality", type=int, default=85, help="Качество для JPEG/WEBP (1-100)"
    )
    parser.add_argument(
        "--subsampling",
        type=parse_subsampling,
        default="auto",
        help="Субсемплинг для JPEG: auto (по умолчанию), keep, 0(4:4:4), 1(4:2:0), 2(4:2:2)",
    )
    parser.add_argument(
        "--png-level", type=int, default=6, help="PNG compress_level (0-9)"
    )
    parser.add_argument(
        "--keep-exif",
        action="store_true",
        default=True,
        help="Сохранять EXIF/ICC (по умолчанию True)",
    )
    parser.add_argument(
        "--no-keep-exif",
        action="store_false",
        dest="keep_exif",
        help="Не сохранять EXIF/ICC",
    )
    parser.add_argument(
        "--convert-to-jpeg",
        action="store_true",
        help="Конвертировать непрозрачные PNG/TIFF/WEBP в JPEG",
    )
    parser.add_argument(
        "--jpeg-bg",
        type=parse_bg,
        default=None,
        help="Цвет фона для заливки прозрачности при конвертации в JPEG, напр. '#ffffff'",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Всегда использовать сжатую версию, даже если она больше оригинала",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Количество потоков обработки",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Только посчитать, без сохранения"
    )

    args = parser.parse_args()

    src_root = args.src.resolve()
    if not src_root.exists() or not src_root.is_dir():
        print(f"[ERR] Папка не найдена: {src_root}", file=sys.stderr)
        sys.exit(1)

    if args.max_size and args.max_size <= 0:
        args.max_size = None

    if args.overwrite:
        out_root = src_root
    else:
        out_root = (
            args.out.resolve() if args.out else Path(f"{src_root}_compressed").resolve()
        )

    files = list(iter_image_files(src_root))
    if not files:
        print("[INFO] Подходящих файлов не найдено.")
        sys.exit(0)

    total_before = 0
    total_after = 0
    compressed_cnt = 0
    copied_cnt = 0
    error_cnt = 0

    print(f"[INFO] Файлов: {len(files)}; вывод: {out_root}")
    if not args.overwrite and out_root.exists() and not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        total_before = sum(p.stat().st_size for p in files)
        print(
            f"[DRY] Всего файлов: {len(files)}; общий объём: {human_bytes(total_before)}"
        )
        print("[DRY] Для точной оценки размера после сжатия запустите без --dry-run.")
        sys.exit(0)

    def choose_jpeg_subsampling(
        original_fmt: str, save_fmt: str
    ) -> Optional[Union[int, str]]:
        if save_fmt.upper() not in ("JPEG", "JPG"):
            return None
        if isinstance(args.subsampling, int):
            return args.subsampling
        # auto/keep
        if str(args.subsampling).lower() in ("keep", "auto"):
            return "keep" if original_fmt.upper() in ("JPEG", "JPG") else 1
        return 1

    with futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = []
        for src in files:
            futs.append(
                ex.submit(
                    process_one, src, src_root, out_root, args, choose_jpeg_subsampling
                )
            )

        for t in futures.as_completed(futs):
            rel, before, after, used_compressed, note = t.result()
            total_before += before
            total_after += after
            if "ERROR" in note:
                error_cnt += 1
                print(f"[ERR] {rel} :: {note}")
            elif used_compressed:
                compressed_cnt += 1
                print(
                    f"[OK ] {rel} :: {human_bytes(before)} → {human_bytes(after)} {('(' + note + ')') if note else ''}"
                )
            else:
                copied_cnt += 1
                print(f"[SKP] {rel} :: без выигрыша, взят оригинал")

    saved = total_before - total_after
    ratio = (1 - (total_after / total_before)) * 100 if total_before > 0 else 0.0
    print("\n===== ИТОГО =====")
    print(f"Файлов обработано:  {len(files)}")
    print(f"Сжато:              {compressed_cnt}")
    print(f"Без выигрыша:       {copied_cnt}")
    print(f"Ошибок:             {error_cnt}")
    print(f"Было:               {human_bytes(total_before)}")
    print(f"Стало:              {human_bytes(total_after)}")
    print(f"Экономия:           {human_bytes(saved)} ({ratio:.1f}%)")


def process_one(
    src: Path,
    src_root: Path,
    out_root: Path,
    args: argparse.Namespace,
    choose_subsampling_cb,
) -> Tuple[str, int, int, bool, str]:
    rel = src.relative_to(src_root)
    dst = src if args.overwrite else (out_root / rel)
    if not args.overwrite and args.suffix:
        dst = dst.with_name(dst.stem + args.suffix + dst.suffix)

    note = ""
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            original_format = (im.format or src.suffix.lstrip(".")).upper()

            if args.max_size:
                max_side = int(args.max_size)
                if max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.LANCZOS)

            save_fmt = original_format

            if original_format in ("HEIC", "HEIF"):
                save_fmt = "JPEG"
                im = im.convert("RGB")
                note = "heif→jpeg"

            if args.convert_to_jpeg and save_fmt not in ("JPEG", "JPG"):
                if not has_alpha(im):
                    im = im.convert("RGB")
                    save_fmt = "JPEG"
                    note = (note + "," if note else "") + "to-jpeg"
                elif args.jpeg_bg:
                    im = flatten_alpha(im, args.jpeg_bg)
                    save_fmt = "JPEG"
                    note = (note + "," if note else "") + "alpha→bg"

            chosen_subsampling = choose_subsampling_cb(original_format, save_fmt)

            used_compressed, new_bytes, orig_bytes = safe_save(
                img=im,
                out_path=dst,
                original_path=src,
                fmt=save_fmt,
                keep_exif=args.keep_exif,
                quality=args.quality,
                subsampling=chosen_subsampling,
                png_level=args.png_level,
                skip_if_larger=not args.force,
            )
            return (str(rel), orig_bytes, new_bytes, used_compressed, note)

    except Exception as e:
        if not args.overwrite:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst)
        size = src.stat().st_size if src.exists() else 0
        return (str(rel), size, size, False, f"ERROR: {e}")


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 12000 * 12000
    main()
