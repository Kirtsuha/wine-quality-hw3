# Приложение для подбора вин

Проверено с Python 3.11.1. Точные версии библиотек находятся в requirements.txt.
Модель не обучается в приложении: она сохраняется вместе с масштабированием из ноутбука.
Порог выбран на валидации: максимум precision при recall >= 0.7.
Дубликаты исходных строк удалены до разбиения на train и test.

## Локальный запуск

Откройте терминал в папке репозитория и выполните:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

В папке должны находиться app.py, model.joblib, config.json и test.csv.
Для CSV-загрузки используйте 12 столбцов из test.csv без target.
Значения должны быть числами без пропусков, is_red — 0 или 1.

## Кэширование

st.cache_resource сохраняет один общий объект модели на сервере.
st.cache_data сохраняет таблицы и массив вероятностей, возвращая копии результатов.
Аргумент _model не хешируется. Вероятности на test считаются один раз для неизменных данных;
ползунок меняет только сравнение с порогом. Без декоратора joblib.load читал бы модель
с диска после каждого действия пользователя.

## Публикация в Streamlit Community Cloud

Репозиторий: https://github.com/Kirtsuha/wine-quality-hw3

- Ветка: `main`
- Main file path: `app.py`
- Python: `3.11` (Advanced settings)

Создайте приложение на https://share.streamlit.io/ и укажите эти параметры.
Репозиторий приватный: Streamlit должен иметь к нему доступ через ваш GitHub-аккаунт.
Репозиторий содержит только файлы приложения. Деплой выполняется отдельно.

Документация кэширования: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource и https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
