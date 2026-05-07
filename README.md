# Projekt sys-wizjiv2 - ALPR

## Skad sa modele

Modele uzyte w projekcie sa gotowymi modelami z zewnetrznych bibliotek. Nie
byly trenowane od zera w tym projekcie.

Model `Python/models/yolov8n.pt` jest gotowym, pobranym modelem YOLOv8n od
Ultralytics. W aplikacji sluzy do wykrywania pojazdow, m.in. klas `car`,
`truck`, `bus` i `motorcycle`.

Oficjalne zrodlo:

- GitHub Ultralytics: https://github.com/ultralytics/ultralytics
- Gotowe wagi YOLOv8n: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

Rozpoznawanie tablic korzysta z biblioteki `fast-alpr`. Ta biblioteka sklada
caly pipeline ALPR z dwoch gotowych czesci:

- detekcja tablic: `open-image-models`,
- OCR, czyli odczyt tekstu tablicy: `fast-plate-ocr`.

Oficjalne zrodla:

- FastALPR: https://github.com/ankandrew/fast-alpr
- modele detekcji tablic `open-image-models`: https://github.com/ankandrew/open-image-models
- modele OCR tablic `fast-plate-ocr`: https://github.com/ankandrew/fast-plate-ocr
- model zoo OCR: https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/

W projekcie sa ustawione gotowe modele z tych bibliotek:

- `yolo-v9-t-384-license-plate-end2end`
- `european-plates-mobile-vit-v2-model`

`yolo-v9-t-384-license-plate-end2end` to gotowy model do wykrywania obszaru
tablicy rejestracyjnej. Pochodzi z `open-image-models`.

`european-plates-mobile-vit-v2-model` to gotowy model OCR do odczytu tekstu
europejskich tablic rejestracyjnych. Wedlug dokumentacji `fast-plate-ocr` jest
to model wytrenowany na europejskich tablicach z ponad 40 krajow i na ponad
40 tysiacach tablic. Ten model nie byl trenowany w tym projekcie, tylko jest
uzywany jako gotowy model z biblioteki.

## Opis projektu

Projekt jest aplikacja do automatycznego rozpoznawania tablic rejestracyjnych
na obrazie, filmie lub przechwyconym ekranie. Aplikacja wykrywa tablice,
odczytuje ich tekst, zapisuje historie wykryc i pokazuje prosta analize
wynikow.

## Funkcje

- analiza plikow wideo i obrazow,
- analiza przechwyconego ekranu,
- rysowanie ramek wokol pojazdow i tablic,
- wypisywanie odczytanego numeru tablicy nad ramka,
- lista obserwowanych tablic w `Python/list.txt` z edycja z poziomu zakladki `Watchlista`,
- zakladka `Analiza` z historia, rankingiem tablic, regionami i trafieniami z watchlisty,
- zakladka `Szukaj` do filtrowania historii po rejestracji i ogladania zdjec,
- profile pracy `CPU Lite` i `GPU Heavy`,
- monitor pracy aplikacji z PID procesu Pythona, zuzyciem CPU, RAM i GPU.

## Uruchomienie

Na Linuxie:

```bash
cd Python
./run_ai.sh
```

Po uruchomieniu aplikacja jest dostepna pod adresem:

```text
http://127.0.0.1:8000
```

Skrypt tworzy lokalne srodowisko `venv`, instaluje zaleznosci z
`Python/requirements.txt` i uruchamia serwer FastAPI przez `uvicorn`.

## Profile pracy

`CPU Lite` jest profilem lekkim. Wysyla do analizy mniejsze klatki, analizuje
rzadziej i nie uruchamia pelnego renderowania filmu w tle. Ten tryb jest
przeznaczony do komputerow bez mocnej karty graficznej.

`GPU Heavy` jest profilem mocniejszym. Probuje uzyc CUDA, analizuje czesciej,
wlacza wykrywanie/sledzenie pojazdow i tworzy pelny wynikowy film z ramkami.

## Najwazniejsze pliki

- `Python/app/web.py` - interfejs WWW, monitor, profile CPU/GPU, watchlista i endpointy,
- `Python/app/pipeline.py` - glowna logika analizy obrazu i rysowania ramek,
- `Python/app/models_runtime.py` - ladowanie modeli ALPR i YOLO,
- `Python/app/config.py` - konfiguracja progow, modeli i profili pracy,
- `Python/models/yolov8n.pt` - gotowy model YOLOv8n,
- `Python/list.txt` - lista tablic oznaczanych jako niebezpieczne.
