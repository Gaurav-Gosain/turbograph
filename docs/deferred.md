# Deferred work (draft)

Items consciously deferred during the O3 / O10 / stemmer-parameter work.
Each entry records the intent, so future sessions can pick it up without
re-deriving the context. None of these are scheduled.

## 1. Пайплайн как единый источник правды (код + GUI)

Сегодня пайплайн retrieval зашит в двух ручных копиях:

- `rag.Store.Retrieve` (`rag/store.go`) — фиксированная цепочка вызовов:
  dense → BM25+RRF → (опц. PPR) → (опц. entity) → MMR → rerank → min-sim.
  `ScoreComponents` («comps») вручную зеркалит ту же математику.
- `PIPE` (`server/static/index.html`) — ручная JS-копия тех же стадий с
  привязкой к локальным настройкам (`tg.settings`).
  Часть стадий живёт только в chat-пути (`rerank`, `min_sim`, `window`,
  `verify`), `/api/query` их не принимает — редактор этого не показывает.

Новый сигнал = правки в N местах (Retrieve, rel/comps, PIPE, API-структуры).

Целевое состояние:
1. Серверный манифест стадий `GET /api/pipeline`: имя, тип
   (always/toggle/range/num), мин/макс/дефолт, маппинг на параметр,
   признак query-vs-chat. PIPE строится из манифеста, а не из кода.
2. Реестр скора-контрибьюторов: новый сигнал добавляется регистрацией,
   а не правкой Retrieve и comps.
3. Единый источник данных для редактора: настройки из манифеста,
   а не разъезжающиеся localStorage-списки.
4. Решить, показывать ли в PIPE различие query/chat-путей, и как.

## 2. Пересмотр фильтров в пайплайне / GUI

Сейчас фильтрация только мета-предикат AND `k=v` (`meta_filter` в
`/api/query` и `/api/chat`, hook `RetrieveParams.MetaFilter`, документный
уровень) — сделано минимально по O10. Отложено:

- Синтаксис шире `k=v`: JSON-path / предикат-выражение (AND сейчас —
  осознанный минимум, согласован с формой `--meta` при инжесте).
- Фильтры по полям документа, по `Chunk.Kind` (image/text), по doc subset;
  размещение pre (в сбор кандидатов) vs post (по top-k) — сейчас
  фильтр применяется к кандидат-сету до скоринга, это и есть «post-only
  на top-k» по смыслу O10, но семантику стоит зафиксировать в доке.
- UI-стадия фильтра: маппинг на PIPE-этап, запись в RuntimeConfig.
- Как фильтр сочетается с PageRank-сидами и entity-сигналом.

## 3. O3 — дополнительно к инкрементальным коммьюнити

Сделано ядро: `graph.UpdateCommunities` (стабильные метки, локальная
label propagation на affected-выборку), аппенд-путь в `rebuildGraph`,
суммаризации переживают add. Отложено:

- Ключевые слова коммьюнити через DF-IDF вместо LLM-суммаризации
  (дёшево, даёт подпись «о чём кластер»).
- UI: показ пустых и раздутых коммьюнити (O3 текст).
- Обработка «устаревания» суммаризаций: стабильные метки не инвалидируют
  `commSummary`, но состав членов может меняться — нужна политика
  (пересобирать только затронутые, помечать stale).

## 4. Стэн-параметр: расширения поверх сделанного

Сделано: `stemmer.Select(langs)`, `lexical.Config.StemLangs`,
`rag.Config.LexicalStemLangs`, CLI `--stem-langs`, серверные
`RuntimeConfig.stem / stem_langs` с live-ретокенизацией открытых бакетов,
поле в UI-конфиге (не стадия PIPE — индекс и запрос обязаны совпадать
по токенизации). Отложено:

- GUI-поле в манифест-стадиях при переезде на единый источник правды
  (пункт 1): показать «index built with stemming/langs» read-only-бейджами.
- Проброс `early_exit` и `ef/ef_gather` в API/редактор (сейчас только
  build-time: `rag.Config.EfGather`, `EarlyExitThreshold`, per-query
  `RetrieveParams.EarlyExit` — в веб-API не выведен). По умолчанию off.
- Query-embedding LRU — намеренно не юзер-кноб (инфраструктура процесса).

## 5. e2e-проверка редактора

Автотестов на рендер HTML нет: изменения покрываются go-тестами
эндпоинтов, но саму страницу браузером после правок конфига (Config →
stem/stem_langs) надо прогнать вручную / через browser-harness:
открыть флоу, переключить стем-параметр, пере-инжестить, проверить
query/chat с meta_filter.

## How to pick up

Current code line anchors (may drift):
- `rag/store.go` — `RetrieveParams` (строки ~906), `seedScores` (~1249),
  точки применения фильтра (~1014, 1141, 1153), `rebuildGraph` (~843).
- `graph/community.go` — `DetectCommunities`, `UpdateCommunities`.
- `stemmer/stemmer.go` — `Register`, `Select`, `StemToken`.
- `server/config.go` — `RuntimeConfig` (GET/POST `/api/config`).
- `server/web.go` — `chatRequest` (~440), `handlePull`.
- `server/static/index.html` — `PIPE` (~2439), конфиг-секция.