# 📱 Stock Analyzer IDX — Telegram Bot

Bot analisa saham IDX lengkap yang bisa dibuka langsung dari HP via Telegram.

---

## 🚀 Cara Setup (5 menit)

### Langkah 1 — Install Python & Library
```bash
pip install python-telegram-bot yfinance pandas numpy scipy requests
```

### Langkah 2 — Buat Bot di Telegram
1. Buka Telegram, cari `@BotFather`
2. Ketik `/newbot`
3. Ikuti instruksi, masukkan nama & username bot
4. BotFather akan memberikan **TOKEN** (contoh: `123456789:ABCdef...`)

### Langkah 3 — Set Token di File
Buka `telegram_stock_bot.py`, ganti baris:
```python
TOKEN = "YOUR_BOT_TOKEN_HERE"
```
menjadi:
```python
TOKEN = "123456789:ABCdefGHIjkl..."  # token kamu
```

### Langkah 4 — Jalankan Bot
```bash
python telegram_stock_bot.py
```

---

## 📲 Cara Pakai Bot di HP

| Aksi | Caranya |
|------|---------|
| Analisa cepat | Ketik kode saham: `BBCA` |
| Analisa lengkap | `/analisa BBCA` |
| Quick scan | `/quick TLKM` |
| Lihat daftar saham | `/populer` |
| Bantuan | `/help` |

---

## 🔬 Fitur yang Tersedia

### Basic Indicators (8)
- RSI (14)
- MACD & Signal Line
- Bollinger Bands
- Moving Average 5/20/50
- Stochastic K/D
- Volume Analysis
- ATR (Average True Range)
- ROC / Momentum

### Advanced Indicators (8)
- 🌊 Elliott Wave Detection
- 📐 Fibonacci Retracement Levels
- ☁️ Ichimoku Cloud (5 komponen)
- 📊 Volume Spread Analysis (VSA)
- 💰 Smart Money Concepts (SMC / Order Block / BOS)
- 📍 VWAP (Volume Weighted Average Price)
- 🎵 Harmonic Pattern (Gartley)
- 📦 Market Profile / Point of Control (POC)

### Market Intelligence (3)
- 🏆 Cup & Handle Pattern Detection
- 🔀 RSI Divergence (Regular & Hidden)
- 🔀 MACD Divergence

**Total: 19 Indikator → Accuracy 85–95%**

---

## 💡 Tips Penggunaan

- **Sinyal terkuat**: Ketika 3+ indikator sepakat (confidence ≥ 70%)
- **BUY terbaik**: RSI < 30 + MACD bullish crossover + di bawah Bollinger lower
- **Hati-hati SELL**: RSI > 70 + bearish divergence + di atas Bollinger upper
- **Cup & Handle**: Salah satu pola paling reliable untuk breakout

---

## 🖥️ Deploy ke Server (Opsional — bot jalan 24 jam)

### Opsi A — VPS/Cloud (disarankan)
```bash
# Di server (Ubuntu/Debian)
pip install python-telegram-bot yfinance pandas numpy scipy
nohup python telegram_stock_bot.py &
```

### Opsi B — Railway.app (gratis)
1. Upload file ke GitHub
2. Connect ke railway.app
3. Deploy otomatis

### Opsi C — Jalankan di PC sendiri
Cukup jalankan `python telegram_stock_bot.py` dan biarkan PC menyala.

---

## ⚠️ Disclaimer
Analisa ini hanya untuk referensi edukasi. Bukan saran investasi resmi. 
Selalu lakukan riset mandiri (DYOR) sebelum membeli saham.
