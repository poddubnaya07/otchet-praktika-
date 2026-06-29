# Сравнительный анализ архитектур для подсчёта людей в толпе

## Описание проекта
Исследование пяти архитектур (YOLOv8n, YOLOv8x, YOLOv5nu, RT-DETR, HOG+SVM) для задачи Crowd Counting.

## Результаты
- Лучшая точность: RT-DETR (2.0 чел.)
- Самая быстрая: YOLOv5nu (20.3 мс)
- Лучший баланс: YOLOv8n (conf=0.1)

## Запуск
1. Установить зависимости: `pip install -r requirements.txt`
2. Открыть `notebooks/experiment.ipynb` в Kaggle
3. Запустить `demo/app.py` для Gradio интерфейса

## Структура
- `notebooks/` - Jupyter ноутбук
- `src/` - исходный код
- `data/` - тестовые изображения
- `runs/` - результаты
- `demo/` - Gradio приложение
- `report/` - отчет по практике