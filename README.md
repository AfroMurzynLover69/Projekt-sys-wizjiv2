# Projekt sys-wizjiv2 - ALPR

Projekt jest aplikacja do automatycznego rozpoznawania tablic rejestracyjnych
na obrazie, filmie lub przechwyconym ekranie. Aplikacja wykrywa tablice,
odczytuje ich tekst, zapisuje historie wykryc i pokazuje prosta analize
wynikow.

## Funkcje

- analiza plikow wideo i obrazow,
- analiza przechwyconego ekranu,
- rysowanie ramek wokol pojazdow i tablic,
- wypisywanie odczytanego numeru tablicy nad ramka,
- lista obserwowanych tablic w `Python/list.txt`,
- zakladka `Analiza` z historia, rankingiem tablic, regionami i trafieniami z watchlisty,
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

## Skad jest model

Model `Python/models/yolov8n.pt` jest gotowym, pobranym modelem YOLOv8n od
Ultralytics. Nie byl trenowany od zera w tym projekcie. W aplikacji sluzy do
wykrywania pojazdow, m.in. klas `car`, `truck`, `bus` i `motorcycle`.

Oficjalne zrodlo:

- GitHub Ultralytics: https://github.com/ultralytics/ultralytics
- Gotowe wagi YOLOv8n: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

Rozpoznawanie tablic i OCR korzysta z biblioteki `fast-alpr` oraz gotowych
modeli skonfigurowanych w `Python/app/config.py`:

- `yolo-v9-t-384-license-plate-end2end`
- `european-plates-mobile-vit-v2-model`

## Najwazniejsze pliki

- `Python/app/web.py` - interfejs WWW, monitor, profile CPU/GPU i endpointy,
- `Python/app/pipeline.py` - glowna logika analizy obrazu i rysowania ramek,
- `Python/app/models_runtime.py` - ladowanie modeli ALPR i YOLO,
- `Python/app/config.py` - konfiguracja progow, modeli i profili pracy,
- `Python/models/yolov8n.pt` - gotowy model YOLOv8n,
- `Python/list.txt` - lista tablic oznaczanych jako niebezpieczne.
