# Projekt sys-wizjiv2 - ALPR

## Modele i zrodla

Projekt uzywa gotowych modeli. Nie byly trenowane w tym repozytorium.

- Tablice rejestracyjne i OCR: `fast-alpr` od `ankandrew`
  - detekcja tablic: `yolo-v9-t-384-license-plate-end2end`
  - odczyt tekstu: `european-plates-mobile-vit-v2-model`
  - glowne zrodlo: https://github.com/ankandrew/fast-alpr
  - zrodla:
    - https://github.com/ankandrew/open-image-models
    - https://github.com/ankandrew/fast-plate-ocr
    - https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/

- Pojazdy: Ultralytics YOLO
  - CPU Lite: `Python/models/yolov8n.pt`
  - GPU Heavy: `Python/models/yolo11x.pt`
  - fallback: jezeli `yolo11x.pt` nie dziala albo go nie ma, kod moze uzyc `yolov8n.pt`
  - zrodla:
    - https://github.com/ultralytics/ultralytics
    - https://github.com/ultralytics/assets/releases

YOLO sluzy tutaj do wykrywania pojazdow (`car`, `truck`, `bus`, `motorcycle`).
FastALPR sluzy do wykrywania tablic i odczytu numeru rejestracyjnego.

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

`CPU Lite` uzywa FastALPR do tablic i OCR oraz `yolov8n.pt` do pojazdow na CPU.
Analizuje rzadziej i jest przeznaczony do slabszych komputerow.

`GPU Heavy` uzywa FastALPR do tablic i OCR oraz `yolo11x.pt` do pojazdow na GPU.
Analizuje czesciej i tworzy pelny wynikowy film z ramkami.

## Najwazniejsze pliki

- `Python/app/web.py` - interfejs WWW, monitor, profile CPU/GPU, watchlista i endpointy,
- `Python/app/pipeline.py` - glowna logika analizy obrazu i rysowania ramek,
- `Python/app/models_runtime.py` - ladowanie modeli ALPR i YOLO,
- `Python/app/config.py` - konfiguracja progow, modeli i profili pracy,
- `Python/models/yolov8n.pt` - gotowy model YOLOv8n,
- `Python/models/yolo11x.pt` - gotowy model YOLO11x,
- `Python/list.txt` - lista tablic oznaczanych jako niebezpieczne.
