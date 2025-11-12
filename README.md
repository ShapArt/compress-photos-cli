# 🗜️ compress-photos-cli

**Ключевые факты:**
- 🗜️ Массовое сжатие, сохранение EXIF/ICC
- 🔄 HEIC→JPEG, указание максимально допустимой стороны
- ⚙️ Параллельная обработка


<table>
<tr>
<td><b>✨ Что умеет</b><br/>Короткий список возможностей, ориентированных на ценность.</td>
<td><b>🧠 Технологии</b><br/>Стек, ключевые решения, нюансы безопасности.</td>
<td><b>🖼️ Демо</b><br/>Скриншот/гиф или ссылка на Pages.</td>
</tr>
</table>

> [!TIP]
> Репозиторий оформлен по правилам: Conventional Commits, SemVer, CHANGELOG, SECURITY policy и CI.
> Секреты — только через `.env`/секреты репозитория.


<p align="left">
  <img alt="build" src="https://img.shields.io/github/actions/workflow/status/ShapArt/compress-photos-cli/ci.yml?label=CI&logo=githubactions">
  <img alt="license" src="https://img.shields.io/github/license/ShapArt/compress-photos-cli">
  <img alt="last commit" src="https://img.shields.io/github/last-commit/ShapArt/compress-photos-cli">
  <img alt="issues" src="https://img.shields.io/github/issues/ShapArt/compress-photos-cli">
  <img alt="stars" src="https://img.shields.io/github/stars/ShapArt/compress-photos-cli?style=social">
</p>


Рекурсивное сжатие фото с EXIF/ICC, конвертацией HEIC → JPEG и ограничением длинной стороны.

## Примеры
```bash
python compress_photos.py ./input --max-size 2560 --quality 84 --workers 8
python compress_photos.py ./input --out ./output_min --convert-to-jpeg --max-size 3000
python compress_photos.py ./input --overwrite --suffix ""
```

## Быстрый старт

*Заполнить по мере развития проекта.*


## Архитектура

*Заполнить по мере развития проекта.*


## Конфигурация

*Заполнить по мере развития проекта.*


## Тесты

*Заполнить по мере развития проекта.*


## Roadmap

*Заполнить по мере развития проекта.*
