# E-Forecast

[![CI](https://github.com/denizefekaracakaya/forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/denizefekaracakaya/forecast/actions/workflows/ci.yml)

Prophet tabanlı günlük hisse fiyat tahmin platformu. Streamlit arayüzü geçmiş fiyatları, güven
aralıklı tahminleri, temel verileri ve duygu analizli haberleri bir arada sunar; FastAPI gateway
fiyat/haber API'si, kullanıcı hesapları ve kişiselleştirilmiş haber akışı sağlar.

> **Uyarı:** Yatırım tavsiyesi değildir; yalnızca eğitim amaçlıdır.

## Desteklenen semboller

- **BIST (Türkiye):** 99 hisse (`THYAO.IS`, `ASELS.IS`, `GARAN.IS`, …)
- **ABD (US):** 105 hisse/ETF (`SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, …)

Tam liste `forecast/config.py` içinde ya da çalışan gateway'de `GET /api/v1/markets` ile görülebilir.

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements-dev.txt   # yalnızca çalıştırmak için: requirements.txt
cp .env.example .env                  # değerleri düzenleyin
```

Yerel geliştirmede Docker olmadan çalışmak için `.env` içinde SQLite kullanın:

```
DATABASE_URL=sqlite+aiosqlite:///./forecast.db
JWT_SECRET_KEY=<python -c "import secrets; print(secrets.token_urlsafe(48))" çıktısı>
```

## Çalıştırma

| Yöntem | Komut | Adresler |
| --- | --- | --- |
| Hepsi (yerel) | `python run_local.py` | UI `:8501`, API `:8000`, API dokümanı `:8000/docs` |
| Yalnızca UI | `streamlit run app.py` | `http://localhost:8501` |
| Yalnızca API | `python scripts/run_gateway.py` | `http://localhost:8000/docs` |
| Docker | `docker compose up --build` | PostgreSQL + UI + API + ingestion worker |

Gateway çalışmıyorsa UI tahmin ve haber özelliklerini yine sunar; giriş, izleme listesi ve
kişiselleştirilmiş haberler devre dışı kalır.

## API (özet)

| Uç nokta | Açıklama |
| --- | --- |
| `GET /health` | Sağlık kontrolü + sürüm |
| `GET /api/v1/markets` | Piyasa → sembol listesi |
| `GET /api/v1/prices/{ticker}?start=YYYY-MM-DD&end=YYYY-MM-DD` | Normalize OHLCV (varsayılan pencere `CACHE_TTL_SECONDS` boyunca önbelleklenir) |
| `GET /api/v1/news[/{ticker}]` | Piyasa / sembol haberleri |
| `POST /api/v1/auth/{register,login,verify-email,forgot-password,reset-password,…}` | Hesap işlemleri |
| `GET/PUT /api/v1/users/{id}/preferences`, `…/watchlist`, `…/feedback`, `…/news` | Kişiselleştirme (JWT gerekir) |

Tüm istek gövdeleri doğrulanır; hatalı alanlar `422` döner.

## Güvenlik notları

- `JWT_SECRET_KEY` boş ya da bilinen bir yer tutucuysa gateway rastgele, işlem başına bir anahtar
  kullanır ve uyarı loglar (yeniden başlatmada oturumlar düşer). Üretimde mutlaka ayarlayın.
- Şifreler bcrypt ile, e-posta doğrulama / şifre sıfırlama token'ları SHA-256 özetiyle saklanır.
- Başarısız girişler kimlik başına sınırlanır (`LOGIN_MAX_ATTEMPTS` deneme /
  `LOGIN_LOCKOUT_SECONDS` saniye, ardından `429`).
- Yeni şifreler en az `PASSWORD_MIN_LENGTH` (varsayılan 8) karakter olmalıdır.
- CORS kaynakları `CORS_ORIGINS` ile kısıtlanabilir.

## Test ve lint

```bash
pytest -q          # ~100 test; ağ erişimi gerektirmez
ruff check .
```

GitHub Actions her push/PR'da lint, testler ve Docker imaj derlemesini çalıştırır
(`.github/workflows/ci.yml`).

## Backtest ve hiperparametre tuning

```bash
python scripts/run_backtest.py SPCX --trials 30 --threshold 15
```

Optuna (TPE) ile `--trials` deneme yapılır (varsayılan `OPTUNA_N_TRIALS`), her deneme
`data/experiments.csv` dosyasına yazılır ve en iyi MAPE raporlanır. MAPE eşiği aşılırsa script
exit code 1 döner. Haftalık backtest iş akışı (`.github/workflows/backtest.yml`) sonucu CSV'yi
artefakt olarak yükler.

## Model

- **Algoritma:** Prophet (`prophet` paketi)
- **Regresörler:** hacim, hacim değişimi, 7 günlük hareketli ortalama, RSI, 5 günlük momentum, SMA20/SMA50 kesişimi
- **Değerlendirme:** `cross_validation` + naif baz çizgisi (dünkü kapanış = yarın)
- **Varsayılan hiperparametreler:** `changepoint_prior_scale=0.05`, `seasonality_prior_scale=10.0`, `seasonality_mode=multiplicative`

Günlük hisse tahmini doğası gereği zordur; model çoğu zaman naif baz çizgisinden anlamlı şekilde
daha iyi olmayabilir.

## Proje yapısı

```
app.py                      # Streamlit arayüzü
run_local.py                # UI + gateway'i birlikte başlatır
forecast/
├── config.py               # Ayarlar (.env), sembol listeleri
├── i18n.py                 # TR/EN çeviriler
├── data/                   # Fiyat sağlayıcıları, normalizasyon, önbellek, haber + NLP, ORM, e-posta
├── forecasting/            # Prophet modeli, backtest, Optuna tuning
├── gateway/                # FastAPI uygulaması, şemalar, güvenlik, auth/kişiselleştirme rotaları
├── ingestion/              # Periyodik veri çekme işçisi
├── personalization/        # Haber sıralama
└── templates/email/        # E-posta şablonları
scripts/                    # run_backtest.py, run_gateway.py, run_ingestion.py
tests/                      # pytest paketi
```

## Streamlit Community Cloud ile deploy

1. Bu repoyu GitHub'a push edin.
2. [share.streamlit.io](https://share.streamlit.io) üzerinden **New app** oluşturun.
3. Repository, branch ve main file path olarak `app.py` seçin.
4. **Deploy** — yalnızca tahmin/haber özellikleri için ekstra secret gerekmez.

## Notlar

- Yahoo Finance sık istekte geçici rate limit uygular. Sağlayıcı üstel geri çekilmeli
  `YAHOO_MAX_RETRIES` deneme yapar; sorun sürerse birkaç dakika bekleyin.
- Ingestion işçisinin önbelleği şimdilik işlem içidir (gateway ile paylaşılmaz); paylaşımlı bir
  arka uç (ör. Redis) yol haritasındadır.
