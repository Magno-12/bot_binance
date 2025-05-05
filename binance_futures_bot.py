# binance_futures_bot.py
import pandas as pd
import numpy as np
import time
from binance.client import Client
from binance.enums import *
import logging
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv
from binance_bot import IndicatorCalculator, SignalGenerator, PerformanceTracker

# Configuración de logging
logger = logging.getLogger("binance_futures_bot")

class TradeGuaranteeManager:
    """Administra la garantía de operaciones mínimas diarias"""
    
    def __init__(self, min_daily_trades=3):
        self.min_daily_trades = min_daily_trades
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()
        self.forced_trade_threshold = 0.75
        
    def reset_daily_count(self):
        """Resetea el contador diario"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades = []
            self.last_reset_date = current_date
            return True
        return False
    
    def register_trade(self, trade_type):
        """Registra una nueva operación"""
        self.daily_trades.append({
            'timestamp': datetime.now(),
            'type': trade_type,
            'was_forced': False
        })
    
    def needs_forced_trades(self):
        """Determina si necesitamos forzar operaciones"""
        self.reset_daily_count()
        
        current_hour = datetime.now().hour
        day_progress = current_hour / 24.0
        
        if day_progress >= self.forced_trade_threshold:
            needed_trades = self.min_daily_trades - len(self.daily_trades)
            return max(0, needed_trades)
        return 0
    
    def get_daily_trades_count(self):
        """Obtiene el conteo de operaciones del día"""
        self.reset_daily_count()
        return len(self.daily_trades)

class BinanceFuturesBot:
    """Clase para trading de futuros en Binance"""

    def __init__(self, symbol="ETHUSDT", interval=Client.KLINE_INTERVAL_1HOUR, params=None, testnet=True):
        """Inicializa el bot de futuros"""
        self.symbol = symbol
        self.interval = interval
        self.testnet = testnet

        # Cargar API keys
        load_dotenv()
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise ValueError("Las claves API no están configuradas en el archivo .env")

        # Mantener el intervalo de 5 minutos
        self.interval = "5m"

        # Parámetros con mayor sensibilidad para generar más de 2 señales diarias
        self.params = {
            # Bandas de Bollinger mucho más sensibles
            'LENGTH_BB': 5,            # Reducido a mínimo para máxima sensibilidad
            'MULT_BB': 1.0,            # Bandas más estrechas para más señales

            # Medias móviles más cortas para mejor respuesta
            'LENGTH_SMA': 10,          # Muy corto para responder a cada movimiento
            'LENGTH_EMA': 10,          # Reducido para responder más rápido

            # RSI con umbrales más amplios
            'LENGTH_RSI': 5,           # RSI muy corto para oscilaciones rápidas
            'RSI_OVERSOLD': 50,        # Punto medio, genera señales constantemente
            'RSI_OVERBOUGHT': 50,      # Umbral más bajo para más señales de venta

            # MACD ultra sensible
            'LENGTH_MACD_SHORT': 3,    # Ultra corto para cruces frecuentes
            'LENGTH_MACD_LONG': 6,     # Diferencia mínima para cruces constantes
            'LENGTH_MACD_SIGNAL': 3,   # Señal muy rápida

            # Gestión de riesgo para operaciones más frecuentes
            'STOP_LOSS_PERCENT': 0.5,  # Stop loss muy ajustado
            'TAKE_PROFIT_PERCENT': 1.0, # Objetivos más pequeños pero más frecuentes

            # Configuración de operaciones
            'ENABLE_SHORT': True,
            'LEVERAGE': 5,
            'HEDGE_MODE': False,
            'EQUITY_PERCENTAGE': 5,     # Ligeramente reducido para distribuir el riesgo

            'RSI_OVERSOLD': 45,      # Añadir explícitamente 
            'RSI_OVERBOUGHT': 55,    # Añadir explícitamente
            'LOOKBACK_PERIOD': 200,
        }

        # Sobrescribir con parámetros personalizados si se proporcionan
        if params:
            self.params.update(params)

        # Inicializar cliente de Binance para futuros
        if self.testnet:
            self.client = Client(self.api_key, self.api_secret, testnet=True)
            self.futures_client = self.client  # En Client reciente, el mismo objeto maneja futuros
            logger.info("Bot de futuros configurado en modo TEST (testnet)")
        else:
            self.client = Client(self.api_key, self.api_secret)
            self.futures_client = self.client
            logger.info("Bot de futuros configurado en modo PRODUCCIÓN")
        
        # Estado del bot
        self.in_position = False
        self.position_side = None  # 'LONG', 'SHORT', o 'BOTH' en modo de cobertura
        self.entry_price = 0
        self.position_amount = 0
        self.stop_loss_order_id = None
        self.take_profit_order_id = None
        
        # Tracker de rendimiento
        self.tracker = PerformanceTracker(data_file='futures_performance_data.json')

        self.trade_guarantee = TradeGuaranteeManager(min_daily_trades=3)
        
        # Configuración inicial
        self._setup_account()
        
        logger.info(f"Bot de futuros inicializado para {self.symbol} en intervalo {self.interval}")
    
    def _setup_account(self):
        """Configura la cuenta de futuros con los parámetros iniciales"""
        try:
            # Verificar si la cuenta está en modo de cobertura (hedge)
            account_info = self.futures_client.futures_account()
            current_mode = 'Hedge' if account_info.get('dualSidePosition') else 'One-way'
            
            # Configurar modo de posición según parámetros
            target_mode = 'Hedge' if self.params['HEDGE_MODE'] else 'One-way'
            if current_mode != target_mode:
                self.futures_client.futures_change_position_mode(
                    dualSidePosition=self.params['HEDGE_MODE']
                )
                logger.info(f"Modo de posición cambiado a: {target_mode}")
            
            # Configurar apalancamiento
            leverage_info = self.futures_client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.params['LEVERAGE']
            )
            logger.info(f"Apalancamiento configurado: {leverage_info.get('leverage')}x")
            
            # Configurar tipo de margen (aislado o cruzado)
            # Por defecto usaremos margen aislado para mayor seguridad
            self.futures_client.futures_change_margin_type(
                symbol=self.symbol,
                marginType='ISOLATED'
            )
            logger.info("Tipo de margen configurado: ISOLATED")
            
        except Exception as e:
            logger.error(f"Error al configurar cuenta de futuros: {e}")
            if "No need to change position side" in str(e):
                logger.info(f"La cuenta ya está en modo {target_mode}")
            elif "No need to change margin type" in str(e):
                logger.info("La cuenta ya está en margen ISOLATED")
            else:
                raise
    
    def get_historical_klines(self):
        """Obtiene datos históricos desde Binance para futuros"""
        try:
            # Calcular fecha de inicio basada en el período de lookback
            start_time = (datetime.now() - timedelta(days=self.params['LOOKBACK_PERIOD'] // 24)).strftime("%d %b %Y %H:%M:%S")
            
            # Obtener klines de futuros
            klines = self.futures_client.futures_klines(
                symbol=self.symbol,
                interval=self.interval,
                startTime=None,  # Binance API traduce el start_time adecuadamente
                limit=self.params['LOOKBACK_PERIOD'] + 100  # Pedir más datos para asegurar suficientes
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
            logger.error(f"Error al obtener datos históricos de futuros: {e}")
            return None
    
    def get_account_balance(self):
        """Obtiene el balance de USDT en la cuenta de futuros"""
        try:
            # Obtener balance de futuros
            account = self.futures_client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
            return 0
        except Exception as e:
            logger.error(f"Error al obtener balance de futuros: {e}")
            return 0
    
    def get_position_info(self):
        """Obtiene información actual de las posiciones abiertas"""
        try:
            positions = self.futures_client.futures_position_information(symbol=self.symbol)
            
            if self.params['HEDGE_MODE']:
                # En modo hedge, habrá una entrada para LONG y otra para SHORT
                position_data = {
                    'LONG': None,
                    'SHORT': None
                }
                
                for position in positions:
                    if position['symbol'] == self.symbol:
                        if position['positionSide'] == 'LONG':
                            position_data['LONG'] = position
                        elif position['positionSide'] == 'SHORT':
                            position_data['SHORT'] = position
                
                return position_data
            else:
                # En modo one-way, solo habrá una entrada para el símbolo
                for position in positions:
                    if position['symbol'] == self.symbol:
                        position_amount = float(position['positionAmt'])
                        if position_amount > 0:
                            return {'LONG': position, 'SHORT': None}
                        elif position_amount < 0:
                            return {'LONG': None, 'SHORT': position}
                        else:
                            return {'LONG': None, 'SHORT': None}
                
                return {'LONG': None, 'SHORT': None}
                
        except Exception as e:
            logger.error(f"Error al obtener información de posición: {e}")
            return {'LONG': None, 'SHORT': None}
    
    def calculate_position_size(self, risk_factor=1.0):
        """Calcula el tamaño de la posición basado en el porcentaje del capital y factor de riesgo"""
        balance = self.get_account_balance()
        
        # Calcular tamaño de la posición con apalancamiento
        position_size = (balance * self.params['EQUITY_PERCENTAGE'] / 100) * self.params['LEVERAGE']
        
        # Ajustar según factor de riesgo dinámico
        position_size *= risk_factor
        
        # Obtener el precio actual
        price_info = self.futures_client.futures_mark_price(symbol=self.symbol)
        current_price = float(price_info['markPrice'])
        
        # Calcular cantidad de contratos
        contract_amount = position_size / current_price
        
        # Redondear la cantidad según las reglas de Binance
        symbol_info = self.futures_client.futures_exchange_info()
        step_size = None
        
        for symbol_data in symbol_info['symbols']:
            if symbol_data['symbol'] == self.symbol:
                for filter_data in symbol_data['filters']:
                    if filter_data['filterType'] == 'LOT_SIZE':
                        step_size = float(filter_data['stepSize'])
                        break
                break
        
        if step_size:
            # Calcular precisión para el redondeo
            precision = 0
            if step_size < 1:
                precision = len(str(step_size).split('.')[1].rstrip('0'))
            
            # Redondear a la baja al múltiplo más cercano de step_size
            contract_amount = round(contract_amount - (contract_amount % step_size), precision)
        
        return contract_amount
    
    def place_futures_order(self, side, position_side, quantity, order_type=ORDER_TYPE_MARKET, price=None, 
                            stop_price=None, reduce_only=False, close_position=False):
        """
        Coloca una orden en el mercado de futuros
        
        Args:
            side: 'BUY' o 'SELL'
            position_side: 'LONG', 'SHORT' o 'BOTH' (solo en hedge mode)
            quantity: cantidad de contratos
            order_type: tipo de orden (MARKET, LIMIT, STOP, etc.)
            price: precio para órdenes límite
            stop_price: precio para órdenes stop
            reduce_only: si la orden solo debe reducir posiciones existentes
            close_position: si la orden debe cerrar toda la posición
        """
        try:
            params = {
                'symbol': self.symbol,
                'side': side,
                'type': order_type,
                'reduceOnly': reduce_only
            }
            
            # Agregar positionSide solo en modo hedge
            if self.params['HEDGE_MODE'] and position_side != 'BOTH':
                params['positionSide'] = position_side
            
            # Si es orden de cierre completo
            if close_position:
                params['closePosition'] = True
            else:
                params['quantity'] = quantity
            
            # Agregar precio para órdenes límite
            if order_type in [ORDER_TYPE_LIMIT, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_TAKE_PROFIT_LIMIT]:
                params['price'] = price
                params['timeInForce'] = TIME_IN_FORCE_GTC
            
            # Agregar stop price para órdenes stop
            if order_type in [ORDER_TYPE_STOP_LOSS, ORDER_TYPE_STOP_LOSS_LIMIT, 
                            ORDER_TYPE_TAKE_PROFIT, ORDER_TYPE_TAKE_PROFIT_LIMIT]:
                params['stopPrice'] = stop_price
            
            # Ejecutar la orden
            response = self.futures_client.futures_create_order(**params)
            
            logger.info(f"Orden de futuros colocada: {side} {quantity} {self.symbol} - ID: {response['orderId']}")
            return response
            
        except Exception as e:
            logger.error(f"Error al colocar orden de futuros: {e}")
            return None
    
    def place_long_position(self, risk_factor=1.0):
        """Abre una posición larga en futuros"""
        if self.in_position and self.position_side in ['LONG', 'BOTH'] and not self.params['HEDGE_MODE']:
            logger.info("Ya hay una posición abierta, no se puede abrir otra posición larga")
            return
        
        try:
            # Calcular cantidad a comprar con gestión de riesgo dinámica
            quantity = self.calculate_position_size(risk_factor)
            
            if quantity <= 0:
                logger.warning("La cantidad a comprar es demasiado pequeña")
                return
            
            # En modo hedge, definir el lado de posición
            position_side = 'LONG' if self.params['HEDGE_MODE'] else 'BOTH'
            
            # Ejecutar la orden de compra
            order = self.place_futures_order(
                side=SIDE_BUY,
                position_side=position_side,
                quantity=quantity,
                order_type=ORDER_TYPE_MARKET
            )
            
            if not order:
                logger.error("Error al colocar orden de apertura larga")
                return
            
            # Obtener precio de entrada (precio de ejecución)
            # Esperar un momento para que la orden se procese
            time.sleep(1)
            order_info = self.futures_client.futures_get_order(
                symbol=self.symbol,
                orderId=order['orderId']
            )
            
            entry_price = float(order_info.get('avgPrice', 0))
            if entry_price == 0:
                # Si no tenemos el precio promedio, usar el precio actual
                price_info = self.futures_client.futures_mark_price(symbol=self.symbol)
                entry_price = float(price_info['markPrice'])
            
            # Actualizar estado
            self.in_position = True
            self.position_side = position_side
            self.entry_price = entry_price
            self.position_amount = quantity
            
            # Iniciar seguimiento en el tracker de rendimiento
            self.tracker.start_trade(
                entry_price=entry_price,
                side='LONG',
                quantity=quantity
            )
            
            logger.info(f"Posición LARGA abierta: Precio={entry_price}, Cantidad={quantity}")
            
            # Colocar órdenes de stop loss y take profit
            self.place_stop_loss_take_profit()
            
            return order
            
        except Exception as e:
            logger.error(f"Error al abrir posición larga: {e}")
            return None
    
    def place_short_position(self, risk_factor=1.0):
        """Abre una posición corta en futuros"""
        if self.in_position and self.position_side in ['SHORT', 'BOTH'] and not self.params['HEDGE_MODE']:
            logger.info("Ya hay una posición abierta, no se puede abrir otra posición corta")
            return
        
        try:
            # Calcular cantidad a vender con gestión de riesgo dinámica
            quantity = self.calculate_position_size(risk_factor)
            
            if quantity <= 0:
                logger.warning("La cantidad a vender es demasiado pequeña")
                return
            
            # En modo hedge, definir el lado de posición
            position_side = 'SHORT' if self.params['HEDGE_MODE'] else 'BOTH'
            
            # Ejecutar la orden de venta
            order = self.place_futures_order(
                side=SIDE_SELL,
                position_side=position_side,
                quantity=quantity,
                order_type=ORDER_TYPE_MARKET
            )
            
            if not order:
                logger.error("Error al colocar orden de apertura corta")
                return
            
            # Obtener precio de entrada (precio de ejecución)
            # Esperar un momento para que la orden se procese
            time.sleep(1)
            order_info = self.futures_client.futures_get_order(
                symbol=self.symbol,
                orderId=order['orderId']
            )
            
            entry_price = float(order_info.get('avgPrice', 0))
            if entry_price == 0:
                # Si no tenemos el precio promedio, usar el precio actual
                price_info = self.futures_client.futures_mark_price(symbol=self.symbol)
                entry_price = float(price_info['markPrice'])
            
            # Actualizar estado
            self.in_position = True
            self.position_side = position_side
            self.entry_price = entry_price
            self.position_amount = quantity
            
            # Iniciar seguimiento en el tracker de rendimiento
            self.tracker.start_trade(
                entry_price=entry_price,
                side='SHORT',
                quantity=quantity
            )
            
            logger.info(f"Posición CORTA abierta: Precio={entry_price}, Cantidad={quantity}")
            
            # Colocar órdenes de stop loss y take profit
            self.place_stop_loss_take_profit()
            
            return order
            
        except Exception as e:
            logger.error(f"Error al abrir posición corta: {e}")
            return None
    
    def place_stop_loss_take_profit(self):
        """Coloca órdenes de stop loss y take profit para la posición actual"""
        try:
            # Cancelar órdenes SL/TP anteriores si existen
            if self.stop_loss_order_id or self.take_profit_order_id:
                self.cancel_all_open_orders()
            
            # Calcular precios de SL y TP según el lado de la posición
            if self.position_side in ['LONG', 'BOTH'] and self.position_amount > 0:
                stop_loss_price = self.entry_price * (1 - self.params['STOP_LOSS_PERCENT'] / 100)
                take_profit_price = self.entry_price * (1 + self.params['TAKE_PROFIT_PERCENT'] / 100)
                
                # Lado de la orden para cerrar (opuesto a la posición)
                close_side = SIDE_SELL
                
            elif self.position_side in ['SHORT', 'BOTH'] and self.position_amount > 0:
                stop_loss_price = self.entry_price * (1 + self.params['STOP_LOSS_PERCENT'] / 100)
                take_profit_price = self.entry_price * (1 - self.params['TAKE_PROFIT_PERCENT'] / 100)
                
                # Lado de la orden para cerrar (opuesto a la posición)
                close_side = SIDE_BUY
                
            else:
                logger.error("No hay posición activa para colocar SL/TP")
                return
            
            # Redondear precios
            stop_loss_price = round(stop_loss_price, 2)  # Ajustar según la precisión del símbolo
            take_profit_price = round(take_profit_price, 2)  # Ajustar según la precisión del símbolo
            
            # Si está activado el trailing stop, usar el tipo de orden adecuado
            if self.params['TRAILING_STOP']:
                # Para trailing stop, necesitamos calcular la distancia de activación
                # basada en la callback rate (ejemplo: 1% significa que el precio 
                # debe retroceder 1% desde el máximo/mínimo alcanzado)
                callback_rate = self.params['TRAILING_STOP_CALLBACK']
                
                # Colocar stop loss con trailing stop
                sl_order = self.futures_client.futures_create_order(
                    symbol=self.symbol,
                    side=close_side,
                    type='TRAILING_STOP_MARKET',
                    quantity=self.position_amount,
                    callbackRate=callback_rate,
                    reduceOnly=True,  # Asegurar que solo cierra posiciones existentes
                    positionSide=self.position_side if self.params['HEDGE_MODE'] else 'BOTH'
                )
                
                self.stop_loss_order_id = sl_order['orderId']
                logger.info(f"Trailing Stop Loss colocado: callback rate {callback_rate}%")
                
            else:
                # Colocar stop loss normal
                sl_order = self.place_futures_order(
                    side=close_side,
                    position_side=self.position_side,
                    quantity=self.position_amount,
                    order_type=ORDER_TYPE_STOP_MARKET,
                    stop_price=stop_loss_price,
                    reduce_only=True
                )
                
                if sl_order:
                    self.stop_loss_order_id = sl_order['orderId']
                    logger.info(f"Stop Loss colocado: {stop_loss_price}")
            
            # Colocar take profit
            tp_order = self.place_futures_order(
                side=close_side,
                position_side=self.position_side,
                quantity=self.position_amount,
                order_type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                stop_price=take_profit_price,
                reduce_only=True
            )
            
            if tp_order:
                self.take_profit_order_id = tp_order['orderId']
                logger.info(f"Take Profit colocado: {take_profit_price}")
            
        except Exception as e:
            logger.error(f"Error al colocar órdenes de SL/TP: {e}")
    
    def cancel_all_open_orders(self):
        """Cancela todas las órdenes abiertas para el símbolo actual"""
        try:
            result = self.futures_client.futures_cancel_all_open_orders(symbol=self.symbol)
            logger.info(f"Todas las órdenes abiertas canceladas: {result}")
            
            # Resetear IDs de órdenes
            self.stop_loss_order_id = None
            self.take_profit_order_id = None
            
            return result
        except Exception as e:
            logger.error(f"Error al cancelar órdenes abiertas: {e}")
            return None
    
    def close_position(self, position_side=None):
        """Cierra la posición actual o la especificada en position_side"""
        try:
            # Si no se especifica lado de posición, cerrar la posición actual
            if position_side is None:
                position_side = self.position_side
            
            # Si no hay posición, no hacer nada
            if not self.in_position:
                logger.info("No hay posición abierta para cerrar")
                return False
            
            # Cancelar órdenes existentes primero
            self.cancel_all_open_orders()
            
            # Obtener información actualizada de la posición
            position_info = self.get_position_info()
            
            # En modo hedge, cerrar el lado específico
            if self.params['HEDGE_MODE']:
                if position_side == 'LONG' and position_info['LONG']:
                    pos_amount = abs(float(position_info['LONG']['positionAmt']))
                    if pos_amount > 0:
                        order = self.place_futures_order(
                            side=SIDE_SELL,
                            position_side='LONG',
                            quantity=pos_amount,
                            reduce_only=True
                        )
                        
                elif position_side == 'SHORT' and position_info['SHORT']:
                    pos_amount = abs(float(position_info['SHORT']['positionAmt']))
                    if pos_amount > 0:
                        order = self.place_futures_order(
                            side=SIDE_BUY,
                            position_side='SHORT',
                            quantity=pos_amount,
                            reduce_only=True
                        )
                
                else:
                    logger.warning(f"No se encontró posición {position_side} para cerrar")
                    return False
            
            # En modo one-way, cerrar toda la posición
            else:
                # Determinar lado de cierre basado en la posición
                if position_info['LONG']:
                    # Posición larga, necesitamos vender
                    order = self.place_futures_order(
                        side=SIDE_SELL,
                        position_side='BOTH',
                        quantity=0,  # No importa, usamos closePosition
                        close_position=True
                    )
                    
                elif position_info['SHORT']:
                    # Posición corta, necesitamos comprar
                    order = self.place_futures_order(
                        side=SIDE_BUY,
                        position_side='BOTH',
                        quantity=0,  # No importa, usamos closePosition
                        close_position=True
                    )
                    
                else:
                    logger.warning("No se encontró posición para cerrar")
                    return False
            
            # Esperar confirmación y obtener precio de cierre
            time.sleep(1)
            
            # Obtener precio actual para el P&L
            price_info = self.futures_client.futures_mark_price(symbol=self.symbol)
            current_price = float(price_info['markPrice'])
            
            # Registrar el cierre en el tracker
            if self.tracker.current_trade:
                self.tracker.end_trade(
                    exit_price=current_price,
                    reason="MANUAL_CLOSE"
                )
            
            # Actualizar estado
            self.in_position = False
            self.position_side = None
            self.entry_price = 0
            self.position_amount = 0
            
            logger.info(f"Posición cerrada a precio aproximado de {current_price}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al cerrar posición: {e}")
            return False
    
    def check_position_status(self):
        """Verifica el estado de la posición actual y actualiza si es necesario"""
        if not self.in_position:
            return
        
        try:
            # Obtener información actualizada de la posición
            position_info = self.get_position_info()
            
            # Verificar si aún tenemos una posición abierta
            has_position = False
            
            if self.params['HEDGE_MODE']:
                if self.position_side == 'LONG' and position_info['LONG']:
                    pos_amount = float(position_info['LONG']['positionAmt'])
                    has_position = pos_amount > 0
                    if has_position:
                        current_price = float(position_info['LONG']['markPrice'])
                        unrealized_pnl = float(position_info['LONG']['unrealizedProfit'])
                        
                elif self.position_side == 'SHORT' and position_info['SHORT']:
                    pos_amount = float(position_info['SHORT']['positionAmt'])
                    has_position = pos_amount < 0  # Negativo para SHORT
                    if has_position:
                        current_price = float(position_info['SHORT']['markPrice'])
                        unrealized_pnl = float(position_info['SHORT']['unrealizedProfit'])
            else:
                # En modo one-way
                if position_info['LONG']:
                    has_position = float(position_info['LONG']['positionAmt']) > 0
                    if has_position:
                        current_price = float(position_info['LONG']['markPrice'])
                        unrealized_pnl = float(position_info['LONG']['unrealizedProfit'])
                        
                elif position_info['SHORT']:
                    has_position = float(position_info['SHORT']['positionAmt']) < 0
                    if has_position:
                        current_price = float(position_info['SHORT']['markPrice'])
                        unrealized_pnl = float(position_info['SHORT']['unrealizedProfit'])
            
            # Si ya no tenemos posición pero el bot cree que sí, actualizar el estado
            if not has_position:
                # La posición fue cerrada (probablemente por SL/TP)
                if self.tracker.current_trade:
                    # Obtener precio actual
                    price_info = self.futures_client.futures_mark_price(symbol=self.symbol)
                    exit_price = float(price_info['markPrice'])
                    
                    # Determinar razón probable de salida
                    reason = "UNKNOWN"
                    if self.entry_price > 0:
                        if self.position_side in ['LONG', 'BOTH'] and exit_price <= self.entry_price * (1 - self.params['STOP_LOSS_PERCENT'] / 100):
                            reason = "STOP_LOSS"
                        elif self.position_side in ['LONG', 'BOTH'] and exit_price >= self.entry_price * (1 + self.params['TAKE_PROFIT_PERCENT'] / 100):
                            reason = "TAKE_PROFIT"
                        elif self.position_side == 'SHORT' and exit_price >= self.entry_price * (1 + self.params['STOP_LOSS_PERCENT'] / 100):
                            reason = "STOP_LOSS"
                        elif self.position_side == 'SHORT' and exit_price <= self.entry_price * (1 - self.params['TAKE_PROFIT_PERCENT'] / 100):
                            reason = "TAKE_PROFIT"
                    
                    # Registrar cierre en el tracker
                    self.tracker.end_trade(
                        exit_price=exit_price,
                        reason=reason
                    )
                
                # Resetear estado del bot
                self.in_position = False
                self.position_side = None
                self.entry_price = 0
                self.position_amount = 0
                self.stop_loss_order_id = None
                self.take_profit_order_id = None
                
                logger.info("Posición cerrada detectada. Estado del bot actualizado.")
                return
            
            # Si tenemos posición activa, mostrar P&L actual
            if has_position:
                # Calcular P&L porcentual
                if self.position_side in ['LONG', 'BOTH'] and self.entry_price > 0:
                    pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
                elif self.position_side == 'SHORT' and self.entry_price > 0:
                    pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100
                else:
                    pnl_percent = 0
                
                logger.info(f"Estado de posición {self.position_side}: Entrada: {self.entry_price}, "
                          f"Actual: {current_price}, P&L: {pnl_percent:.2f}% ({unrealized_pnl})")
                
                # Verificar órdenes de SL/TP
                if not self.stop_loss_order_id or not self.take_profit_order_id:
                    # Alguna orden fue ejecutada o cancelada, verificar cuál
                    orders = self.futures_client.futures_get_open_orders(symbol=self.symbol)
                    
                    has_sl = any(order.get('orderId') == self.stop_loss_order_id for order in orders)
                    has_tp = any(order.get('orderId') == self.take_profit_order_id for order in orders)
                    
                    if not has_sl and not has_tp:
                        # Ambas órdenes han desaparecido, verificar si necesitamos recolocarlas
                        self.place_stop_loss_take_profit()
                    
                    elif not has_sl and self.stop_loss_order_id:
                        # Stop loss ejecutado, cerrar posición manualmente
                        logger.info("Stop Loss ejecutado. Cerrando posición manualmente si es necesario.")
                        self.close_position()
                    
                    elif not has_tp and self.take_profit_order_id:
                        # Take profit ejecutado, cerrar posición manualmente
                        logger.info("Take Profit ejecutado. Cerrando posición manualmente si es necesario.")
                        self.close_position()
            
        except Exception as e:
            logger.error(f"Error al verificar estado de posición: {e}")
    
    def run_strategy(self):
        """Ejecuta la estrategia de trading en un bucle continuo"""
        logger.info("Iniciando estrategia de trading de futuros...")
        
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

                needed_trades = self.trade_guarantee.needs_forced_trades()

                if needed_trades > 0:
                    logger.info(f"¡ALERTA! Necesitamos {needed_trades} operaciones para cumplir objetivo diario")
                    self._force_trades(needed_trades)
                    continue
                
                # Si no hay posición abierta, buscar señales de entrada
                if not self.in_position:
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
                        
                        self.place_long_position(risk_factor)

                        self.trade_guarantee.register_trade('LONG')
                        
                    elif sell_signal:
                        logger.info("¡Señal de venta en corto detectada!")
                        
                        # Gestión de riesgo dinámica similar
                        atr_value = df['atr'].iloc[-1]
                        current_price = df['close'].iloc[-1]
                        volatility = atr_value / current_price
                        risk_factor = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
                        
                        self.place_short_position(risk_factor)

                        self.trade_guarantee.register_trade('SHORT')
                
                # Mostrar información relevante
                price_info = self.futures_client.futures_mark_price(symbol=self.symbol)
                current_price = float(price_info['markPrice'])
                
                # Obtener información del libro de órdenes para análisis de liquidez
                order_book = self.futures_client.futures_order_book(symbol=self.symbol, limit=5)
                
                # Calcular el ratio bid/ask (soporte/resistencia inmediata)
                bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
                ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
                bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else 0
                
                logger.info(f"Precio actual: {current_price}, Ratio compra/venta: {bid_ask_ratio:.2f}, "
                          f"En posición: {self.in_position}, Tipo: {self.position_side}")
                
                # Mostrar información de financiamiento (específico de futuros)
                try:
                    funding_info = self.futures_client.futures_funding_rate(symbol=self.symbol)
                    if funding_info:
                        funding_rate = float(funding_info[0].get('fundingRate', 0)) * 100  # Convertir a porcentaje
                        next_funding_time = datetime.fromtimestamp(funding_info[0].get('nextFundingTime', 0) / 1000)
                        logger.info(f"Tasa de financiamiento actual: {funding_rate:.4f}%, "
                                   f"Próximo financiamiento: {next_funding_time}")
                except Exception as e:
                    logger.error(f"Error al obtener información de financiamiento: {e}")
                
                # Esperar para la próxima iteración
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Interrupción manual detectada, finalizando...")
                break
                
            except Exception as e:
                logger.error(f"Error en el ciclo principal: {e}")
                time.sleep(60)
        
        # Cierre controlado
        self._cleanup()

    def _force_trades(self, count):
        """Fuerza operaciones para cumplir el objetivo diario de 3 operaciones"""
        logger.info(f"Iniciando secuencia de {count} operaciones forzadas para cumplir objetivo diario")
        
        for i in range(count):
            try:
                # Si hay posición abierta, cerrarla primero
                if self.in_position:
                    logger.info(f"Cerrando posición actual antes de operación forzada #{i+1}")
                    self.close_position()
                    time.sleep(5)  # Esperar que se cierre la posición
                
                # Alternar entre long y short para operaciones forzadas
                trade_type = 'LONG' if i % 2 == 0 else 'SHORT'
                
                logger.info(f"¡FORZANDO OPERACIÓN #{i+1} DE {count} - TIPO: {trade_type}!")
                
                # Usar factor de riesgo mínimo para operaciones forzadas
                risk_factor = 0.3  # Factor muy bajo para minimizar riesgo
                
                if trade_type == 'LONG':
                    self.place_long_position(risk_factor)
                else:
                    self.place_short_position(risk_factor)
                
                # ¡NUEVO! Registrar la operación como forzada
                if len(self.trade_guarantee.daily_trades) > 0:
                    self.trade_guarantee.daily_trades[-1]['was_forced'] = True
                
                # Esperar un momento entre operaciones forzadas
                time.sleep(15)  # Dar tiempo para que se procese la operación
                
            except Exception as e:
                logger.error(f"Error en operación forzada #{i+1}: {e}")
                time.sleep(10)
    
    def _cleanup(self):
        """Realiza limpieza al finalizar la ejecución"""
        try:
            logger.info("Realizando limpieza final...")
            
            # Cancelar todas las órdenes pendientes
            self.cancel_all_open_orders()
            
            # Cerrar posiciones abiertas
            if self.in_position:
                self.close_position()
            
            # ¡NUEVO! Generar reporte final de operaciones del día
            daily_trades = self.trade_guarantee.daily_trades
            daily_count = len(daily_trades)
            
            if daily_count > 0:
                logger.info(f"Resumen de operaciones del día: {daily_count} operaciones")
                for i, trade in enumerate(daily_trades, 1):
                    forced_status = "FORZADA" if trade.get('was_forced', False) else "NATURAL"
                    logger.info(f"  Operación #{i}: {trade['type']} - {forced_status}")
            
            logger.info("Limpieza finalizada correctamente")
            
        except Exception as e:
            logger.error(f"Error durante la limpieza final: {e}")
    
    def get_funding_rate_history(self, limit=30):
        """Obtiene el historial de tasas de financiamiento para análisis"""
        try:
            funding_history = self.futures_client.futures_funding_rate_history(
                symbol=self.symbol,
                limit=limit
            )
            
            # Convertir a DataFrame para análisis
            df_funding = pd.DataFrame(funding_history)
            df_funding['fundingRate'] = df_funding['fundingRate'].astype(float) * 100  # Convertir a porcentaje
            df_funding['fundingTime'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')
            
            return df_funding
        except Exception as e:
            logger.error(f"Error al obtener historial de tasas de financiamiento: {e}")
            return None
    
    def get_open_interest(self, period="5m"):
        """Obtiene datos de interés abierto para análisis de sentimiento"""
        try:
            open_interest = self.futures_client.futures_open_interest_hist(
                symbol=self.symbol,
                period=period,
                limit=30
            )
            
            # Convertir a DataFrame
            df_oi = pd.DataFrame(open_interest)
            df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
            df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
            df_oi['sumOpenInterestValue'] = df_oi['sumOpenInterestValue'].astype(float)
            
            return df_oi
        except Exception as e:
            logger.error(f"Error al obtener datos de interés abierto: {e}")
            return None
    
    def get_leverage_brackets(self):
        """Obtiene los rangos de apalancamiento disponibles para el símbolo"""
        try:
            brackets = self.futures_client.futures_leverage_bracket(symbol=self.symbol)
            return brackets
        except Exception as e:
            logger.error(f"Error al obtener rangos de apalancamiento: {e}")
            return None
