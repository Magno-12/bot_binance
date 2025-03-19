# binance_bot.py
import pandas as pd
import numpy as np
import time
from binance.client import Client
from binance.enums import *
import ta
import math
import logging
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("binance_bot")

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Configuración de la API de Binance
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
# TEST_MODE = os.environ.get('TEST_MODE', 'True').lower() == 'true'
TEST_MODE = os.environ.get('TEST_MODE', 'False').lower() == 'true'

print(f"API Key: {API_KEY[:5]}...{API_KEY[-5:] if API_KEY else ''}")
print(f"API Secret: {API_SECRET[:5]}...{API_SECRET[-5:] if API_SECRET else ''}")
print(f"Test Mode: {TEST_MODE}")

client = Client(API_KEY, API_SECRET, testnet=TEST_MODE)
try:
    # Intentar una operación simple
    status = client.get_system_status()
    print(f"System status: {status}")
    
    # Intentar obtener datos de cuenta
    account = client.get_account()
    print("Conexión exitosa a la API!")
except Exception as e:
    print(f"Error: {e}")

if not API_KEY or not API_SECRET:
    raise ValueError("Las claves API no están configuradas en el archivo .env")

class IndicatorCalculator:
    """Clase para calcular indicadores técnicos"""
    
    @staticmethod
    def calculate_bollinger_bands(df, length=30, mult=2.0):
        """Calcula las Bandas de Bollinger"""
        df['bb_middle'] = ta.trend.sma_indicator(df['close'], length)
        df['bb_std'] = df['close'].rolling(length).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * mult)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * mult)
        return df
    
    @staticmethod
    def calculate_moving_averages(df, sma_length=50, ema_length=50):
        """Calcula SMA y EMA"""
        df['sma'] = ta.trend.sma_indicator(df['close'], sma_length)
        df['ema'] = ta.trend.ema_indicator(df['close'], ema_length)
        return df
    
    @staticmethod
    def calculate_rsi(df, length=14):
        """Calcula el RSI"""
        df['rsi'] = ta.momentum.rsi(df['close'], length)
        return df
    
    @staticmethod
    def calculate_macd(df, short_length=12, long_length=26, signal_length=9):
        """Calcula el MACD"""
        macd = ta.trend.MACD(
            df['close'], 
            window_slow=long_length, 
            window_fast=short_length, 
            window_sign=signal_length
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        return df
    
    @staticmethod
    def calculate_atr(df, length=14):
        """Calcula el ATR (Average True Range) para gestión dinámica de riesgo"""
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], length
        )
        return df
    
    @staticmethod
    def calculate_all_indicators(df, params):
        """Calcula todos los indicadores técnicos"""
        df = IndicatorCalculator.calculate_bollinger_bands(
            df, params['LENGTH_BB'], params['MULT_BB']
        )
        df = IndicatorCalculator.calculate_moving_averages(
            df, params['LENGTH_SMA'], params['LENGTH_EMA']
        )
        df = IndicatorCalculator.calculate_rsi(df, params['LENGTH_RSI'])
        df = IndicatorCalculator.calculate_macd(
            df, params['LENGTH_MACD_SHORT'], params['LENGTH_MACD_LONG'], params['LENGTH_MACD_SIGNAL']
        )
        df = IndicatorCalculator.calculate_atr(df, 14)  # ATR para gestión de riesgo
        return df


class SignalGenerator:
    """Clase para generar señales de trading"""
    
    @staticmethod
    def check_buy_signal(df, current_idx):
        """Verifica las condiciones de compra (posición larga)"""
        if current_idx < 1 or current_idx >= len(df):
            return False
        
        # Condición 1: Precio cruza por encima de la banda inferior de Bollinger y RSI < 30
        cross_lower_bb = (df['close'].iloc[current_idx-1] <= df['bb_lower'].iloc[current_idx-1] and
                          df['close'].iloc[current_idx] > df['bb_lower'].iloc[current_idx])
        rsi_oversold = df['rsi'].iloc[current_idx] < 30
        
        # Condición 2: MACD cruza por encima de la línea de señal
        macd_crossover = (df['macd'].iloc[current_idx-1] <= df['macd_signal'].iloc[current_idx-1] and
                          df['macd'].iloc[current_idx] > df['macd_signal'].iloc[current_idx])
        
        return (cross_lower_bb and rsi_oversold) or macd_crossover
    
    @staticmethod
    def check_sell_signal(df, current_idx):
        """Verifica las condiciones de venta (cierre de posición larga o apertura de corta)"""
        if current_idx < 1 or current_idx >= len(df):
            return False
        
        # Condición 1: Precio cruza por encima de la banda superior de Bollinger y RSI > 70
        cross_upper_bb = (df['close'].iloc[current_idx-1] <= df['bb_upper'].iloc[current_idx-1] and
                          df['close'].iloc[current_idx] > df['bb_upper'].iloc[current_idx])
        rsi_overbought = df['rsi'].iloc[current_idx] > 70
        
        # Condición 2: MACD cruza por debajo de la línea de señal
        macd_crossunder = (df['macd'].iloc[current_idx-1] >= df['macd_signal'].iloc[current_idx-1] and
                           df['macd'].iloc[current_idx] < df['macd_signal'].iloc[current_idx])
        
        return (cross_upper_bb and rsi_overbought) or macd_crossunder


class PerformanceTracker:
    """Clase para realizar seguimiento del rendimiento del bot"""
    
    def __init__(self, data_file='performance_data.json'):
        self.data_file = data_file
        self.trades = []
        self.current_trade = None
        self.equity_curve = []
        
        # Intentar cargar datos existentes
        self.load_data()
    
    def load_data(self):
        """Carga datos de rendimiento guardados previamente"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.equity_curve = data.get('equity_curve', [])
                    logger.info(f"Datos de rendimiento cargados: {len(self.trades)} operaciones")
        except Exception as e:
            logger.error(f"Error al cargar datos de rendimiento: {e}")
    
    def save_data(self):
        """Guarda los datos de rendimiento en un archivo JSON"""
        try:
            data = {
                'trades': self.trades,
                'equity_curve': self.equity_curve
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info("Datos de rendimiento guardados correctamente")
        except Exception as e:
            logger.error(f"Error al guardar datos de rendimiento: {e}")
    
    def start_trade(self, entry_price, side, quantity, timestamp=None):
        """Inicia el seguimiento de una nueva operación"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.current_trade = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'side': side,  # 'LONG' o 'SHORT'
            'quantity': quantity,
            'exit_time': None,
            'exit_price': None,
            'profit_loss': None,
            'profit_loss_percent': None,
            'duration': None,
            'status': 'OPEN'
        }
        
        logger.info(f"Nueva operación iniciada: {side} a {entry_price}")
    
    def end_trade(self, exit_price, timestamp=None, reason=""):
        """Finaliza el seguimiento de una operación actual"""
        if self.current_trade is None:
            logger.warning("No hay operación actual para finalizar")
            return
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Calcular P&L
        if self.current_trade['side'] == 'LONG':
            profit_loss = (exit_price - self.current_trade['entry_price']) * self.current_trade['quantity']
            profit_loss_percent = ((exit_price / self.current_trade['entry_price']) - 1) * 100
        else:  # 'SHORT'
            profit_loss = (self.current_trade['entry_price'] - exit_price) * self.current_trade['quantity']
            profit_loss_percent = ((self.current_trade['entry_price'] / exit_price) - 1) * 100
        
        # Actualizar la operación actual
        self.current_trade.update({
            'exit_time': timestamp,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'duration': self._calculate_duration(self.current_trade['entry_time'], timestamp),
            'exit_reason': reason,
            'status': 'CLOSED'
        })
        
        # Añadir a la lista de operaciones y resetear la actual
        self.trades.append(self.current_trade)
        logger.info(f"Operación finalizada: P&L {profit_loss_percent:.2f}% ({profit_loss:.6f}), Razón: {reason}")
        
        # Actualizar equity curve
        self._update_equity_curve(self.current_trade)
        
        # Guardar datos
        self.save_data()
        
        self.current_trade = None
    
    def _calculate_duration(self, start_time, end_time):
        """Calcula la duración de una operación en formato legible"""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            return str(duration)
        except Exception:
            return "Desconocido"
    
    def _update_equity_curve(self, trade):
        """Actualiza la curva de equidad después de cada operación"""
        last_equity = 100  # Valor inicial si no hay datos previos
        if self.equity_curve:
            last_equity = self.equity_curve[-1]['equity']
        
        new_equity = last_equity * (1 + trade['profit_loss_percent']/100)
        
        self.equity_curve.append({
            'timestamp': trade['exit_time'],
            'equity': new_equity,
            'trade_id': len(self.trades) - 1
        })
    
    def get_performance_stats(self):
        """Calcula y devuelve estadísticas de rendimiento"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_profit_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Estadísticas básicas
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['profit_loss'] > 0])
        losing_trades = len([t for t in self.trades if t['profit_loss'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L
        profits = [t['profit_loss'] for t in self.trades if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in self.trades if t['profit_loss'] < 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else float('inf')
        total_profit_loss = sum([t['profit_loss'] for t in self.trades])
        
        # Cálculo de drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe Ratio (simplificado)
        returns = [t['profit_loss_percent']/100 for t in self.trades]
        sharpe_ratio = (sum(returns) / len(returns)) / (np.std(returns) if len(returns) > 1 else 1) if returns else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_profit_loss': total_profit_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_max_drawdown(self):
        """Calcula el drawdown máximo en la curva de equidad"""
        if not self.equity_curve:
            return 0
        
        equities = [point['equity'] for point in self.equity_curve]
        max_dd = 0
        peak = equities[0]
        
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def generate_report(self, filename='trading_report.txt'):
        """Genera un informe completo de rendimiento"""
        stats = self.get_performance_stats()
        
        report = [
            "====== INFORME DE RENDIMIENTO DEL BOT DE TRADING ======",
            f"Fecha del informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "--- ESTADÍSTICAS GENERALES ---",
            f"Total de operaciones: {stats['total_trades']}",
            f"Operaciones ganadoras: {stats['winning_trades']} ({stats['win_rate']:.2f}%)",
            f"Operaciones perdedoras: {stats['losing_trades']} ({100-stats['win_rate']:.2f}%)",
            f"Rentabilidad total: {stats['total_profit_loss']:.6f}",
            "",
            "--- MÉTRICAS DE RENDIMIENTO ---",
            f"Ratio de beneficio/pérdida: {stats['profit_factor']:.2f}",
            f"Beneficio promedio: {stats['avg_profit']:.6f}",
            f"Pérdida promedio: {stats['avg_loss']:.6f}",
            f"Drawdown máximo: {stats['max_drawdown']:.2f}%",
            f"Ratio de Sharpe: {stats['sharpe_ratio']:.2f}",
            "",
            "--- ÚLTIMAS 10 OPERACIONES ---"
        ]
        
        # Añadir las últimas 10 operaciones
        recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        for i, trade in enumerate(reversed(recent_trades)):
            report.append(f"{i+1}. {trade['side']} | Entrada: {trade['entry_price']} | Salida: {trade['exit_price']} | "
                        f"P&L: {trade['profit_loss_percent']:.2f}% | Duración: {trade['duration']}")
        
        # Escribir el informe a un archivo
        with open(filename, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Informe de rendimiento generado: {filename}")
        return '\n'.join(report)


class Backtester:
    """Clase para realizar backtesting de la estrategia"""
    
    def __init__(self, client, symbol, interval, params, start_date, end_date=None):
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.params = params
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.tracker = PerformanceTracker(data_file='backtest_results.json')
        
    def get_historical_data(self):
        """Obtiene datos históricos para el backtesting"""
        try:
            logger.info(f"Obteniendo datos históricos desde {self.start_date} hasta {self.end_date}")
            
            klines = self.client.get_historical_klines(
                self.symbol,
                self.interval,
                self.start_date,
                self.end_date
            )
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Configurar timestamp como índice
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Datos obtenidos: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            return None
    
    def run_backtest(self, initial_capital=1000, position_size_percent=10):
        """Ejecuta el backtesting completo sobre los datos históricos"""
        df = self.get_historical_data()
        if df is None or df.empty:
            logger.error("No se pudieron obtener datos para el backtesting")
            return None
        
        # Calcular indicadores
        df = IndicatorCalculator.calculate_all_indicators(df, self.params)
        
        # Variables para el seguimiento del backtest
        current_position = None  # None, 'LONG', o 'SHORT'
        capital = initial_capital
        results = []
        
        # Iterar a través de los datos (empezando después de que todos los indicadores estén calculados)
        start_idx = max(
            self.params['LENGTH_BB'],
            self.params['LENGTH_SMA'],
            self.params['LENGTH_EMA'],
            self.params['LENGTH_RSI'],
            self.params['LENGTH_MACD_LONG'] + self.params['LENGTH_MACD_SIGNAL']
        )
        
        # Añadir una columna para posiciones en el DataFrame
        df['position'] = None
        
        for i in range(start_idx + 1, len(df)):
            date = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Verificar señales
            buy_signal = SignalGenerator.check_buy_signal(df, i)
            sell_signal = SignalGenerator.check_sell_signal(df, i)
            
            # Gestión de posiciones basada en señales
            if current_position is None:  # Sin posición
                if buy_signal:
                    # Calcular tamaño de posición
                    position_size = (capital * position_size_percent / 100) / current_price
                    
                    # Gestión de riesgo dinámica basada en ATR
                    atr_value = df['atr'].iloc[i]
                    risk_factor = min(1.0, 0.02 / (atr_value / current_price))  # Max 2% de riesgo
                    position_size *= risk_factor
                    
                    entry_price = current_price
                    stop_loss = entry_price * (1 - self.params['STOP_LOSS_PERCENT']/100)
                    take_profit = entry_price * (1 + self.params['TAKE_PROFIT_PERCENT']/100)
                    
                    # Registrar la operación
                    self.tracker.start_trade(
                        entry_price=entry_price,
                        side='LONG',
                        quantity=position_size,
                        timestamp=date.isoformat()
                    )
                    
                    current_position = 'LONG'
                    df.at[date, 'position'] = 'LONG'
                    
                elif sell_signal and self.params.get('ENABLE_SHORT', False):
                    # Similar a la compra pero para posiciones cortas
                    position_size = (capital * position_size_percent / 100) / current_price
                    
                    # Gestión de riesgo dinámica
                    atr_value = df['atr'].iloc[i]
                    risk_factor = min(1.0, 0.02 / (atr_value / current_price))
                    position_size *= risk_factor
                    
                    entry_price = current_price
                    stop_loss = entry_price * (1 + self.params['STOP_LOSS_PERCENT']/100)
                    take_profit = entry_price * (1 - self.params['TAKE_PROFIT_PERCENT']/100)
                    
                    self.tracker.start_trade(
                        entry_price=entry_price,
                        side='SHORT',
                        quantity=position_size,
                        timestamp=date.isoformat()
                    )
                    
                    current_position = 'SHORT'
                    df.at[date, 'position'] = 'SHORT'
            
            elif current_position == 'LONG':  # Tenemos posición larga
                # Comprobar stop loss y take profit
                entry_price = self.tracker.current_trade['entry_price']
                stop_loss = entry_price * (1 - self.params['STOP_LOSS_PERCENT']/100)
                take_profit = entry_price * (1 + self.params['TAKE_PROFIT_PERCENT']/100)
                
                if current_price <= stop_loss:
                    # Stop Loss alcanzado
                    self.tracker.end_trade(
                        exit_price=current_price,
                        timestamp=date.isoformat(),
                        reason="STOP_LOSS"
                    )
                    current_position = None
                    df.at[date, 'position'] = None
                    
                elif current_price >= take_profit:
                    # Take Profit alcanzado
                    self.tracker.end_trade(
                        exit_price=current_price,
                        timestamp=date.isoformat(),
                        reason="TAKE_PROFIT"
                    )
                    current_position = None
                    df.at[date, 'position'] = None
                    
                elif sell_signal:
                    # Señal de venta
                    self.tracker.end_trade(
                        exit_price=current_price,
                        timestamp=date.isoformat(),
                        reason="SELL_SIGNAL"
                    )
                    current_position = None
                    df.at[date, 'position'] = None
            
            elif current_position == 'SHORT' and self.params.get('ENABLE_SHORT', False):  # Tenemos posición corta
                # Comprobar stop loss y take profit para posición corta
                entry_price = self.tracker.current_trade['entry_price']
                stop_loss = entry_price * (1 + self.params['STOP_LOSS_PERCENT']/100)
                take_profit = entry_price * (1 - self.params['TAKE_PROFIT_PERCENT']/100)
                
                if current_price >= stop_loss:
                    # Stop Loss alcanzado (para shorts, es cuando el precio sube)
                    self.tracker.end_trade(
                        exit_price=current_price,
                        timestamp=date.isoformat(),
                        reason="STOP_LOSS"
                    )
                    current_position = None
                    df.at[date, 'position'] = None
                    
                elif current_price <= take_profit:
                    # Take Profit alcanzado (para shorts, es cuando el precio baja)
                    self.tracker.end_trade(
                        exit_price=current_price,
                        timestamp=date.isoformat(),
                        reason="TAKE_PROFIT"
                    )
                    current_position = None
                    df.at[date, 'position'] = None
                    
                elif buy_signal:
                    # Señal de compra (cerrar posición corta)
                    self.tracker.end_trade(
                        exit_price=current_price,
                        timestamp=date.isoformat(),
                        reason="BUY_SIGNAL"
                    )
                    current_position = None
                    df.at[date, 'position'] = None
        
        # Cerrar cualquier posición abierta al final del backtest
        if current_position is not None and self.tracker.current_trade is not None:
            self.tracker.end_trade(
                exit_price=df['close'].iloc[-1],
                timestamp=df.index[-1].isoformat(),
                reason="END_OF_BACKTEST"
            )
        
        # Generar y guardar informe
        backtest_report = self.tracker.generate_report(filename='backtest_report.txt')
        
        # Guardar el DataFrame con las posiciones para análisis posterior
        df.to_csv('backtest_data.csv')
        
        return {
            'df': df,
            'stats': self.tracker.get_performance_stats(),
            'report': backtest_report,
            'trades': self.tracker.trades,
            'equity_curve': self.tracker.equity_curve
        }


class BinanceTradingBot:
    """Clase principal del bot de trading"""
    
    def __init__(self, symbol="ETHUSDT", interval=Client.KLINE_INTERVAL_1HOUR, params=None):
        """Inicializa el bot de trading"""
        self.symbol = symbol
        self.interval = interval
        
        # Parámetros por defecto
        self.params = {
            'LENGTH_BB': 30,
            'MULT_BB': 2.0,
            'LENGTH_SMA': 50,
            'LENGTH_EMA': 50,
            'LENGTH_RSI': 14,
            'LENGTH_MACD_SHORT': 12,
            'LENGTH_MACD_LONG': 26,
            'LENGTH_MACD_SIGNAL': 9,
            'STOP_LOSS_PERCENT': 2,
            'TAKE_PROFIT_PERCENT': 4,
            'LOOKBACK_PERIOD': 200,
            'EQUITY_PERCENTAGE': 10,
            'ENABLE_SHORT': False  # Por defecto, solo posiciones largas
        }
        
        # Sobrescribir con parámetros personalizados si se proporcionan
        if params:
            self.params.update(params)
        
        # Inicializar cliente de Binance (testnet o producción)
        if TEST_MODE:
            self.client = Client(API_KEY, API_SECRET, testnet=True)
            logger.info("Bot configurado en modo TEST (testnet)")
        else:
            self.client = Client(API_KEY, API_SECRET)
            logger.info("Bot configurado en modo PRODUCCIÓN")
        
        # Estado del bot
        self.in_position = False
        self.position_side = None  # 'LONG' o 'SHORT'
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Tracker de rendimiento
        self.tracker = PerformanceTracker()
        
        logger.info(f"Bot inicializado para {self.symbol} en intervalo {self.interval}")
    
    def get_historical_klines(self):
        """Obtiene datos históricos desde Binance."""
        try:
            # Usar una fecha exacta en lugar de un período relativo
            start_time = (datetime.now() - timedelta(days=10)).strftime("%d %b %Y %H:%M:%S")
            klines = self.client.get_historical_klines(
                self.symbol,
                self.interval,
                start_time
            )

            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Configurar timestamp como índice
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            return None
    
    def get_account_balance(self):
        """Obtiene el balance de USDT en la cuenta."""
        try:
            account = self.client.get_account()
            for asset in account['balances']:
                if asset['asset'] == 'USDT':
                    return float(asset['free'])
            return 0
        except Exception as e:
            logger.error(f"Error al obtener balance: {e}")
            return 0
    
    def calculate_position_size(self, risk_factor=1.0):
        """Calcula el tamaño de la posición basado en el porcentaje del capital y factor de riesgo."""
        balance = self.get_account_balance()
        position_size = (balance * self.params['EQUITY_PERCENTAGE']) / 100
        
        # Ajustar según factor de riesgo dinámico
        position_size *= risk_factor
        
        # Obtener el precio actual
        ticker = self.client.get_symbol_ticker(symbol=self.symbol)
        current_price = float(ticker['price'])
        
        # Calcular cantidad de crypto que se puede comprar
        crypto_amount = position_size / current_price
        
        # Redondear la cantidad según las reglas de Binance
        symbol_info = self.client.get_symbol_info(self.symbol)
        lot_size_filter = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
        step_size = float(lot_size_filter['stepSize'])
        
        # Calcular la precisión necesaria para el tamaño del lote
        precision = int(round(-math.log10(step_size)))
        crypto_amount = round(math.floor(crypto_amount * 10**precision) / 10**precision, precision)
        
        return crypto_amount
    
    def place_buy_order(self, risk_factor=1.0):
        """Coloca una orden de compra en el mercado."""
        if self.in_position:
            logger.info("Ya hay una posición abierta, no se puede comprar")
            return
        
        try:
            # Obtener el precio actual
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Calcular cantidad a comprar con gestión de riesgo dinámica
            quantity = self.calculate_position_size(risk_factor)
            
            if quantity <= 0:
                logger.warning("La cantidad a comprar es demasiado pequeña")
                return
            
            # Ejecutar la orden de compra
            order = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            # Registrar la operación
            self.in_position = True
            self.position_side = 'LONG'
            self.entry_price = current_price
            self.stop_loss = current_price * (1 - self.params['STOP_LOSS_PERCENT']/100)
            self.take_profit = current_price * (1 + self.params['TAKE_PROFIT_PERCENT']/100)
            
            # Iniciar seguimiento en el tracker de rendimiento
            self.tracker.start_trade(
                entry_price=current_price,
                side='LONG',
                quantity=quantity
            )
            
            logger.info(f"COMPRA ejecutada: Precio={current_price}, Cantidad={quantity}")
            logger.info(f"Stop Loss={self.stop_loss}, Take Profit={self.take_profit}")
            
            # Colocar órdenes de stop loss y take profit
            self.place_stop_loss_take_profit(quantity)
            
        except Exception as e:
            logger.error(f"Error al colocar orden de compra: {e}")
    
    def place_sell_short_order(self, risk_factor=1.0):
        """Coloca una orden de venta en corto."""
        if self.in_position or not self.params.get('ENABLE_SHORT', False):
            logger.info("No se puede abrir posición corta")
            return
        
        try:
            # Obtener el precio actual
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Calcular cantidad a vender con gestión de riesgo dinámica
            quantity = self.calculate_position_size(risk_factor)
            
            if quantity <= 0:
                logger.warning("La cantidad a vender en corto es demasiado pequeña")
                return
            
            # Ejecutar la orden de venta en corto (requiere margen habilitado)
            order = self.client.create_margin_order(
                symbol=self.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
                sideEffectType="MARGIN_BUY"  # Para venta en corto
            )
            
            # Registrar la operación
            self.in_position = True
            self.position_side = 'SHORT'
            self.entry_price = current_price
            self.stop_loss = current_price * (1 + self.params['STOP_LOSS_PERCENT']/100)
            self.take_profit = current_price * (1 - self.params['TAKE_PROFIT_PERCENT']/100)
            
            # Iniciar seguimiento en el tracker de rendimiento
            self.tracker.start_trade(
                entry_price=current_price,
                side='SHORT',
                quantity=quantity
            )
            
            logger.info(f"VENTA EN CORTO ejecutada: Precio={current_price}, Cantidad={quantity}")
            logger.info(f"Stop Loss={self.stop_loss}, Take Profit={self.take_profit}")
            
            # Colocar órdenes de stop loss y take profit para posición corta
            self.place_stop_loss_take_profit(quantity)
            
        except Exception as e:
            logger.error(f"Error al colocar orden de venta en corto: {e}")
    
    def place_stop_loss_take_profit(self, quantity):
        """Coloca órdenes de stop loss y take profit."""
        try:
            # Definir side según el tipo de posición
            close_side = SIDE_SELL if self.position_side == 'LONG' else SIDE_BUY
            
            # Colocar stop loss
            stop_loss_order = self.client.create_order(
                symbol=self.symbol,
                side=close_side,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                stopPrice=str(self.stop_loss),
                price=str(self.stop_loss * 0.99 if self.position_side == 'LONG' else self.stop_loss * 1.01)
            )
            
            # Colocar take profit
            take_profit_order = self.client.create_order(
                symbol=self.symbol,
                side=close_side,
                type=ORDER_TYPE_TAKE_PROFIT_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                stopPrice=str(self.take_profit),
                price=str(self.take_profit * 1.01 if self.position_side == 'LONG' else self.take_profit * 0.99)
            )
            
            logger.info(f"Órdenes de SL/TP colocadas: SL={self.stop_loss}, TP={self.take_profit}")
            
        except Exception as e:
            logger.error(f"Error al colocar órdenes de SL/TP: {e}")
    
    def check_open_orders(self):
        """Verifica si hay órdenes abiertas y su estado."""
        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            return len(open_orders) > 0
        except Exception as e:
            logger.error(f"Error al verificar órdenes abiertas: {e}")
            return False
    
    def close_position(self):
        """Cierra la posición actual cancelando todas las órdenes y vendiendo/comprando."""
        if not self.in_position:
            return
        
        try:
            # Cancelar todas las órdenes abiertas
            self.client.cancel_all_orders(symbol=self.symbol)
            
            # Obtener la cantidad actual de la criptomoneda
            if self.position_side == 'LONG':
                # Para posiciones largas
                symbol_base = self.symbol.replace('USDT', '')
                balance = self.client.get_asset_balance(asset=symbol_base)
                quantity = float(balance['free'])
                side = SIDE_SELL
            else:
                # Para posiciones cortas
                # Esto requiere cerrar la posición en margen
                margin_trades = self.client.get_margin_trades(symbol=self.symbol)
                quantity = sum([float(trade['qty']) for trade in margin_trades if trade['isBuyer'] is False])
                side = SIDE_BUY
            
            # Ejecutar la orden de cierre
            if quantity > 0:
                order_type = ORDER_TYPE_MARKET
                order_params = {
                    'symbol': self.symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity
                }
                
                if self.position_side == 'SHORT':
                    # Añadir parámetros específicos para cerrar posición corta
                    order_params['sideEffectType'] = 'AUTO_REPAY'
                    order = self.client.create_margin_order(**order_params)
                else:
                    order = self.client.create_order(**order_params)
                
                # Obtener precio actual para el P&L
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                current_price = float(ticker['price'])
                
                # Registrar el cierre en el tracker
                self.tracker.end_trade(
                    exit_price=current_price,
                    reason="MANUAL_CLOSE"
                )
                
                logger.info(f"Posición cerrada: {side} {quantity} a precio de mercado")
            
            # Resetear variables de posición
            self.in_position = False
            self.position_side = None
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            
        except Exception as e:
            logger.error(f"Error al cerrar posición: {e}")
    
    def check_position_status(self):
        """Verifica el estado de la posición actual y actualiza si es necesario."""
        if not self.in_position:
            return
        
        try:
            # Obtener el precio actual
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Comprobar si se ha alcanzado SL o TP
            if self.position_side == 'LONG':
                if current_price <= self.stop_loss:
                    logger.info(f"Stop Loss alcanzado: {current_price} <= {self.stop_loss}")
                    self.close_position()
                    self.tracker.end_trade(current_price, reason="STOP_LOSS")
                elif current_price >= self.take_profit:
                    logger.info(f"Take Profit alcanzado: {current_price} >= {self.take_profit}")
                    self.close_position()
                    self.tracker.end_trade(current_price, reason="TAKE_PROFIT")
            elif self.position_side == 'SHORT':
                if current_price >= self.stop_loss:
                    logger.info(f"Stop Loss alcanzado (SHORT): {current_price} >= {self.stop_loss}")
                    self.close_position()
                    self.tracker.end_trade(current_price, reason="STOP_LOSS")
                elif current_price <= self.take_profit:
                    logger.info(f"Take Profit alcanzado (SHORT): {current_price} <= {self.take_profit}")
                    self.close_position()
                    self.tracker.end_trade(current_price, reason="TAKE_PROFIT")
            
            # Calcular P&L actual
            if self.position_side == 'LONG':
                profit_loss = ((current_price - self.entry_price) / self.entry_price) * 100
            else:  # 'SHORT'
                profit_loss = ((self.entry_price - current_price) / self.entry_price) * 100
            
            logger.info(f"Estado de posición: {self.position_side}, Entrada: {self.entry_price}, "
                      f"Actual: {current_price}, P&L: {profit_loss:.2f}%")
            
        except Exception as e:
            logger.error(f"Error al verificar estado de posición: {e}")
    
    def run_strategy(self):
        """Ejecuta la estrategia de trading en un bucle continuo."""
        logger.info("Iniciando estrategia de trading...")
        
        while True:
            try:
                # Obtener datos históricos
                df = self.get_historical_klines()
                if df is None or df.empty:
                    logger.warning("No se pudieron obtener datos históricos, reintentando...")
                    time.sleep(60)
                    continue
                
                # Calcular indicadores
                df = IndicatorCalculator.calculate_all_indicators(df, self.params)
                
                # Verificar si hay posiciones abiertas
                self.check_position_status()
                
                # Verificar si hay órdenes abiertas
                has_open_orders = self.check_open_orders()
                
                # Si no hay posición abierta, buscar señales de entrada
                if not self.in_position and not has_open_orders:
                    # Verificar señales en los datos más recientes
                    buy_signal = SignalGenerator.check_buy_signal(df, -1)
                    sell_signal = SignalGenerator.check_sell_signal(df, -1)
                    
                    if buy_signal:
                        logger.info("¡Señal de compra detectada!")
                        
                        # Calcular factor de riesgo dinámico basado en ATR
                        atr_value = df['atr'].iloc[-1]
                        current_price = df['close'].iloc[-1]
                        volatility = atr_value / current_price
                        
                        # Ajustar tamaño de posición inversamente proporcional a la volatilidad
                        risk_factor = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
                        
                        self.place_buy_order(risk_factor)
                        
                    elif sell_signal and self.params.get('ENABLE_SHORT', False):
                        logger.info("¡Señal de venta en corto detectada!")
                        
                        # Gestión de riesgo dinámica similar
                        atr_value = df['atr'].iloc[-1]
                        current_price = df['close'].iloc[-1]
                        volatility = atr_value / current_price
                        risk_factor = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
                        
                        self.place_sell_short_order(risk_factor)
                
                # Mostrar estado actual
                current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
                logger.info(f"Precio actual: {current_price}, En posición: {self.in_position}, Tipo: {self.position_side}")
                
                if self.in_position:
                    if self.position_side == 'LONG':
                        profit_loss = ((current_price - self.entry_price) / self.entry_price) * 100
                    else:  # 'SHORT'
                        profit_loss = ((self.entry_price - current_price) / self.entry_price) * 100
                    logger.info(f"P&L actual: {profit_loss:.2f}%")

                # Esperar para la próxima iteración
                time.sleep(30)

            except Exception as e:
                logger.error(f"Error en el ciclo principal: {e}")
                time.sleep(60)

    def run_backtest(self, start_date, end_date=None, initial_capital=1000):
        """Ejecuta un backtest de la estrategia."""
        backtester = Backtester(
            client=self.client,
            symbol=self.symbol,
            interval=self.interval,
            params=self.params,
            start_date=start_date,
            end_date=end_date
        )

        return backtester.run_backtest(initial_capital=initial_capital)

    def generate_performance_report(self):
        """Genera un informe de rendimiento."""
        return self.tracker.generate_report()
