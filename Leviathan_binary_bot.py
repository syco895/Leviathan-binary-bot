import asyncio
from telegram import Bot
from telegram.constants import ParseMode
from twelvedata import TDClient
import pandas as pd

from Algorithm import Candle, run_signal  # ✅ Use your full logic wrapper

# 🔐 Configuration
TOKEN = '8096922506:AAElAjuqu2A0UFXch0MgSUraGtsJo5ITmW4'  # Replace in production
CHAT_ID = '7740020918'
TWELVE_DATA_API_KEY = '15f9aa56059249869dfbbea2913c2b2d'

# Initialize services
bot = Bot(token=TOKEN)
td = TDClient(apikey=TWELVE_DATA_API_KEY)

# 📈 Fetch live candles from Twelve Data
def get_candles_from_api(symbol: str, interval: str = "1min", count: int = 10):
    try:
        ts = td.time_series(symbol=symbol, interval=interval, outputsize=count)
        df = ts.as_pandas()
        df = df.sort_index()

        candles = []
        for i in range(len(df)):
            row = df.iloc[i]
            candle = Candle(
                open=float(row["open"]),
                close=float(row["close"]),
                high=float(row["high"]),
                low=float(row["low"]),
                volume=float(row.get("volume", 1000.0)),
                atr=abs(row["high"] - row["low"]),
                direction='buy' if row["close"] > row["open"] else 'sell',
                prior_candles=candles[-5:] if i >= 5 else [],
            )
            candles.append(candle)

        return candles

    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return []

# 🤖 Signal loop
async def send_signal():
    print("⏳ Checking signal...")

    candles_1m = get_candles_from_api("EUR/USD", "1min", 10)
    candles_5m = get_candles_from_api("EUR/USD", "5min", 5)
    candles_15m = get_candles_from_api("EUR/USD", "15min", 3)

    if not (candles_1m and candles_5m and candles_15m):
        await bot.send_message(chat_id=CHAT_ID, text="⚠️ Could not fetch market data.", parse_mode=ParseMode.HTML)
        return

    # ✅ Full algorithm run with logging
    signal = run_signal("EUR/USD", candles_1m, candles_5m, candles_15m)

    if signal != "NONE":
        message = f"📊 <b>Signal:</b> <code>{signal}</code>\nPair: EUR/USD\nTimeframe: 1M"
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=ParseMode.HTML)
    else:
        print("📭 No signal this round.")

# 🔁 Loop forever
async def main_loop():
    while True:
        try:
            await send_signal()
        except Exception as e:
            print(f"❌ Bot error: {e}")
        await asyncio.sleep(60)

# 🚀 Launch bot
if __name__ == "__main__":
    asyncio.run(main_loop())
