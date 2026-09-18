-- Начальная настройка базы данных для Stage 1. Бизнес-таблицы будут добавлены
-- после проверки схем источников; заранее выдуманные колонки здесь не создаются.
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;
