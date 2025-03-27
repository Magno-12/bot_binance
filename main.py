# main.py
import sys
import os
import argparse
import logging
import time
import traceback
import signal
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Importar componentes mejorados
from notification_system import NotificationSystem
from data_persistence import DataPersistence
from binance_bot import BinanceTradingBot
from binance_futures_bot import BinanceFuturesBot
from healthcheck import HealthCheckServer

# Configuración mejorada de logging
def setup_logging():
    """Configura el sistema de logging con rotación de archivos"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Archivo de log con fecha
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = f"{log_dir}/bot_{today}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("robust_main")

class RobustTradingBot:
    """Versión robusta del bot de trading con manejo de errores y recuperación"""
    
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging()
        self.running = True
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.error_cooldown = 60  # Segundos de espera tras error
        
        # Inicializar servidor de health check
        render_port = int(os.environ.get('PORT', '8080'))
        self.health_server = HealthCheckServer(port=render_port)
        self.health_server.start()
        self.health_server.update_bot_status({
            'status': 'initializing',
            'symbol': args.symbol,
            'interval': args.interval,
            'mode': 'futures' if args.futures else 'spot'
        })
        
        # Inicializar componentes mejorados
        self.persistence = DataPersistence({
            'data_dir': 'data',
            'backup_dir': 'backups',
            'backup_interval': 3600  # Backup cada hora
        })
        
        self.notifier = NotificationSystem()
        
        # Configurar manejo de señales
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        
        # Inicializar el bot de trading
        self.bot = None  # Se inicializará en start()
        
        self.logger.info("Sistema robusto inicializado")
    
    def handle_shutdown(self, signum, frame):
        """Maneja el apagado controlado al recibir señales"""
        self.logger.info(f"Recibida señal de apagado ({signum})")
        self.running = False
    
    def initialize_bot(self):
        """Inicializa el bot de trading con parámetros mejorados"""
        try:
            # Personalizar parámetros
            params = {
                'ENABLE_SHORT': self.args.enable_short,
                # Otros parámetros pueden ser cargados desde un archivo persistente
            }
            
            # Parámetros adicionales para futuros
            if self.args.futures:
                params.update({
                    'LEVERAGE': self.args.leverage,
                    'HEDGE_MODE': self.args.hedge_mode,
                    'TRAILING_STOP': self.args.trailing_stop,
                })
            
            # Intentar cargar parámetros guardados
            config_file = 'futures_bot_params.json' if self.args.futures else 'bot_params.json'
            saved_params = self.persistence.load_data(config_file)
            if saved_params:
                params.update(saved_params)
                self.logger.info(f"Parámetros cargados desde archivo persistente")
            
            # Inicializar el bot según el modo (spot o futuros)
            if self.args.futures:
                self.bot = BinanceFuturesBot(
                    symbol=self.args.symbol,
                    interval=map_interval(self.args.interval),
                    params=params,
                    testnet=self.args.testnet
                )
                self.logger.info(f"Bot de FUTUROS inicializado para {self.args.symbol} en intervalo {self.args.interval}")
                self.logger.info(f"Configuración: Apalancamiento {params.get('LEVERAGE', 3)}x, Hedge Mode: {params.get('HEDGE_MODE', False)}")
            else:
                self.bot = BinanceTradingBot(
                    symbol=self.args.symbol,
                    interval=map_interval(self.args.interval),
                    params=params
                )
                self.logger.info(f"Bot de SPOT inicializado para {self.args.symbol} en intervalo {self.args.interval}")
            
            # Verificar conectividad
            if not self.check_api_connectivity():
                raise Exception("No se pudo establecer conexión con la API de Binance")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al inicializar el bot: {e}")
            self.notifier.notify_error("Error al inicializar el bot", str(e))
            return False
    
    def check_api_connectivity(self):
        """Verifica la conectividad con la API de Binance"""
        try:
            system_status = self.bot.client.get_system_status()
            if system_status:
                self.logger.info(f"Estado del sistema Binance: {system_status}")
                return True
        except Exception as e:
            self.logger.error(f"Error de conectividad con Binance: {e}")
            return False
    
    def start(self):
        """Inicia el bot con manejo de errores y recuperación"""
        self.logger.info("Iniciando sistema robusto de trading...")
        
        # Actualizar estado
        self.health_server.update_bot_status({
            'status': 'starting',
            'last_trade_time': None,
            'trades_today': 0
        })
        
        # Inicializar bot
        if not self.initialize_bot():
            self.logger.error("No se pudo inicializar el bot. Abortando.")
            self.health_server.update_bot_status({'status': 'failed'})
            return False
        
        # Notificar inicio
        mode = "FUTUROS" if self.args.futures else "SPOT"
        self.notifier.notify_restart(f"Inicio del sistema en modo {mode}")
        
        # Actualizar estado
        self.health_server.update_bot_status({'status': 'running'})
        
        # Registrar hora de inicio para estadísticas diarias
        last_daily_summary = datetime.now()
        last_data_save = datetime.now()
        
        while self.running:
            try:
                # Ejecutar un ciclo del bot
                self._run_bot_cycle()
                
                # Resetear contador de errores tras ciclo exitoso
                if self.consecutive_errors > 0:
                    self.consecutive_errors = 0
                    self.logger.info("Operación normalizada después de errores")
                
                # Verificar si es momento de guardar datos
                now = datetime.now()
                if (now - last_data_save).total_seconds() > 300:  # Cada 5 minutos
                    self._save_bot_state()
                    last_data_save = now
                
                # Verificar si es momento de enviar resumen diario
                if (now - last_daily_summary).total_seconds() > 86400:  # 24 horas
                    self._send_daily_summary()
                    last_daily_summary = now
                
                # Pequeña pausa entre ciclos
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("Interrupción manual detectada")
                self.running = False
                
            except Exception as e:
                self.handle_error(e)
        
        # Limpieza final
        self._cleanup()
        self.logger.info("Sistema detenido correctamente")
        return True
    
    def _run_bot_cycle(self):
        """Ejecuta un ciclo completo del bot"""
        # Obtener datos históricos
        df = self.bot.get_historical_klines()
        if df is None or df.empty:
            self.logger.warning("No se pudieron obtener datos históricos")
            self.health_server.update_bot_status({'status': 'warning', 'warning': 'No data available'})
            return
        
        # Calcular indicadores
        if hasattr(self.bot, 'params') and 'calculate_indicators' in self.bot.params:
            df = self.bot.params['calculate_indicators'](df, self.bot.params)
        else:
            # Usar la clase IndicatorCalculator directamente si está disponible
            from binance_bot import IndicatorCalculator
            df = IndicatorCalculator.calculate_all_indicators(df, self.bot.params)
        
        # Verificar si hay posiciones abiertas
        self.bot.check_position_status()
        
        # Actualizar estado en función del tipo de bot
        if self.args.futures:
            # Para bot de futuros
            position_info = self.bot.get_position_info()
            has_position = position_info.get('LONG') is not None or position_info.get('SHORT') is not None
            
            self.health_server.update_bot_status({
                'is_in_position': has_position,
                'position_side': self.bot.position_side,
                'position_entry_price': self.bot.entry_price if has_position else None,
                'leverage': self.bot.params.get('LEVERAGE', 3),
                'last_price': float(df['close'].iloc[-1]) if not df.empty else None
            })
        else:
            # Para bot de spot
            self.health_server.update_bot_status({
                'is_in_position': self.bot.in_position,
                'position_side': self.bot.position_side if self.bot.in_position else None,
                'position_entry_price': self.bot.entry_price if self.bot.in_position else None,
                'last_price': float(df['close'].iloc[-1]) if not df.empty else None
            })
        
        # Si estamos en el modo de futuros, ejecutar el ciclo específico
        if self.args.futures:
            # El bot de futuros ya verifica las señales en su método check_position_status
            # No necesitamos hacer más aquí, excepto actualizar el estado
            pass
        else:
            # Ejecutar el ciclo para el bot de spot
            self._run_spot_bot_cycle(df)
    
    def _run_spot_bot_cycle(self, df):
        """Ejecuta el ciclo específico para el bot de spot"""
        from binance_bot import SignalGenerator
        
        # Si no hay posición abierta, buscar señales de entrada
        if not self.bot.in_position:
            # Verificar señales en los datos más recientes
            buy_signal = SignalGenerator.check_buy_signal(df, -1)
            sell_signal = SignalGenerator.check_sell_signal(df, -1)
            
            if buy_signal:
                self.logger.info("¡Señal de compra detectada!")
                
                # Calcular factor de riesgo dinámico basado en ATR
                atr_value = df['atr'].iloc[-1]
                current_price = df['close'].iloc[-1]
                volatility = atr_value / current_price
                
                # Ajustar tamaño de posición inversamente proporcional a la volatilidad
                risk_factor = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
                
                self.bot.place_buy_order(risk_factor)
                
            elif sell_signal and self.bot.params.get('ENABLE_SHORT', False):
                self.logger.info("¡Señal de venta en corto detectada!")
                
                # Gestión de riesgo dinámica similar
                atr_value = df['atr'].iloc[-1]
                current_price = df['close'].iloc[-1]
                volatility = atr_value / current_price
                risk_factor = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
                
                self.bot.place_sell_short_order(risk_factor)
    
    def _save_bot_state(self):
        """Guarda el estado actual del bot"""
        try:
            # Guardar estado del rastreador de rendimiento
            if hasattr(self.bot, 'tracker') and hasattr(self.bot.tracker, 'trades'):
                data_file = 'futures_performance_data.json' if self.args.futures else 'performance_data.json'
                data = {
                    'trades': self.bot.tracker.trades,
                    'equity_curve': self.bot.tracker.equity_curve
                }
                self.persistence.save_data(data, data_file)
                
            # Guardar parámetros actuales
            params_file = 'futures_bot_params.json' if self.args.futures else 'bot_params.json'
            self.persistence.save_data(self.bot.params, params_file)
            
            # Guardar estado actual (posición, etc.)
            state_file = 'futures_bot_state.json' if self.args.futures else 'bot_state.json'
            
            if self.args.futures:
                state = {
                    'in_position': self.bot.in_position,
                    'position_side': self.bot.position_side,
                    'entry_price': self.bot.entry_price,
                    'position_amount': self.bot.position_amount,
                    'last_update': datetime.now().isoformat()
                }
            else:
                state = {
                    'in_position': self.bot.in_position,
                    'position_side': self.bot.position_side,
                    'entry_price': self.bot.entry_price,
                    'stop_loss': self.bot.stop_loss,
                    'take_profit': self.bot.take_profit,
                    'last_update': datetime.now().isoformat()
                }
                
            self.persistence.save_data(state, state_file)
            
            self.logger.debug("Estado del bot guardado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error al guardar estado del bot: {e}")
    
    def _send_daily_summary(self):
        """Envía un resumen diario de la actividad del bot"""
        try:
            # Calcular estadísticas del día
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            
            # Filtrar operaciones del último día
            daily_trades = []
            if hasattr(self.bot, 'tracker') and hasattr(self.bot.tracker, 'trades'):
                for trade in self.bot.tracker.trades:
                    trade_time = datetime.fromisoformat(trade['exit_time'])
                    if yesterday <= trade_time <= now:
                        daily_trades.append(trade)
            
            # Calcular estadísticas
            total_trades = len(daily_trades)
            winning_trades = len([t for t in daily_trades if t['profit_loss'] > 0])
            losing_trades = total_trades - winning_trades
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum([t['profit_loss'] for t in daily_trades])
            
            best_trade = max([t['profit_loss_percent'] for t in daily_trades]) if daily_trades else 0
            worst_trade = min([t['profit_loss_percent'] for t in daily_trades]) if daily_trades else 0
            
            # Obtener balance actual
            current_balance = self.bot.get_account_balance()
            
            # Información adicional para futuros
            mode = "FUTUROS" if self.args.futures else "SPOT"
            additional_info = ""
            
            if self.args.futures:
                # Obtener información de tasas de financiamiento
                try:
                    funding_info = self.bot.get_funding_rate_history(limit=1)
                    if funding_info is not None and not funding_info.empty:
                        funding_rate = funding_info['fundingRate'].iloc[0]
                        additional_info = f"\nTasa de financiamiento actual: {funding_rate:.6f}%"
                except Exception as e:
                    self.logger.error(f"Error al obtener tasa de financiamiento para informe: {e}")
            
            # Enviar resumen
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'current_balance': current_balance,
                'mode': mode,
                'additional_info': additional_info
            }
            
            self.notifier.notify_daily_summary(stats)
            self.logger.info("Resumen diario enviado")
            
        except Exception as e:
            self.logger.error(f"Error al enviar resumen diario: {e}")
    
    def handle_error(self, error):
        """Maneja errores durante la ejecución del bot"""
        self.consecutive_errors += 1
        
        # Actualizar estado
        self.health_server.update_bot_status({
            'status': 'error',
            'last_error': str(error),
            'error_count': self.consecutive_errors
        })
        
        # Registrar error
        error_msg = str(error)
        error_traceback = traceback.format_exc()
        self.logger.error(f"Error durante la ejecución: {error_msg}")
        self.logger.error(f"Traceback: {error_traceback}")
        
        # Notificar error si es crítico o recurrente
        if self.consecutive_errors >= 3:
            self.notifier.notify_error(
                f"Error recurrente ({self.consecutive_errors})",
                f"{error_msg}\n\n{error_traceback}"
            )
        
        # Verificar si debemos reiniciar el bot
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.logger.warning(f"Demasiados errores consecutivos ({self.consecutive_errors}), reiniciando bot...")
            self.restart_bot()
        else:
            # Esperar antes de continuar (con backoff exponencial)
            cooldown = self.error_cooldown * (2 ** (self.consecutive_errors - 1))
            cooldown = min(cooldown, 1800)  # Máximo 30 minutos
            self.logger.info(f"Esperando {cooldown} segundos antes de continuar...")
            time.sleep(cooldown)
    
    def restart_bot(self):
        """Reinicia el bot tras errores críticos"""
        self.logger.info("Reiniciando bot...")
        
        try:
            # Cerrar posiciones abiertas si es posible
            if self.bot and self.bot.in_position:
                self.logger.info("Intentando cerrar posiciones abiertas...")
                try:
                    self.bot.close_position()
                except Exception as e:
                    self.logger.error(f"Error al cerrar posiciones: {e}")
            
            # Reinicializar el bot
            self.bot = None
            time.sleep(5)  # Pequeña pausa antes de reiniciar
            
            if self.initialize_bot():
                self.logger.info("Bot reiniciado correctamente")
                mode = "FUTUROS" if self.args.futures else "SPOT"
                self.notifier.notify_restart(f"Reinicio tras errores en modo {mode}")
                self.consecutive_errors = 0
            else:
                self.logger.error("No se pudo reiniciar el bot")
                self.running = False  # Detener ejecución
                
        except Exception as e:
            self.logger.error(f"Error durante el reinicio: {e}")
            self.running = False  # Detener ejecución
    
    def _cleanup(self):
        """Realiza limpieza al finalizar la ejecución"""
        try:
            # Actualizar estado
            self.health_server.update_bot_status({'status': 'stopping'})
            
            # Guardar estado final
            self._save_bot_state()
            
            # Cerrar posiciones abiertas si es necesario
            if self.bot and self.bot.in_position:
                self.logger.info("Cerrando posiciones abiertas durante apagado...")
                try:
                    self.bot.close_position()
                except Exception as e:
                    self.logger.error(f"Error al cerrar posiciones durante apagado: {e}")
            
            # Detener sistema de persistencia
            if self.persistence:
                self.persistence.stop()
            
            # Notificar apagado
            mode = "FUTUROS" if hasattr(self.args, 'futures') and self.args.futures else "SPOT"
            self.notifier.notify_error(f"Bot detenido ({mode})", "El sistema ha sido detenido correctamente.")
            
            # Detener servidor de health check
            self.health_server.stop()
            
            self.logger.info("Limpieza finalizada correctamente")
            
        except Exception as e:
            self.logger.error(f"Error durante la limpieza final: {e}")
            
            # Intentar detener health server aunque haya error
            try:
                self.health_server.stop()
            except:
                pass


def parse_args():
    """Analiza los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Bot de Trading de Criptomonedas para Binance (versión robusta)')
    
    # Comando principal
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: run (ejecutar bot en tiempo real)
    run_parser = subparsers.add_parser('run', help='Ejecutar el bot en tiempo real')
    run_parser.add_argument('--symbol', default='ETHUSDT', help='Par de trading (por defecto: ETHUSDT)')
    run_parser.add_argument('--interval', default='1h', help='Intervalo de tiempo (por defecto: 1h)')
    run_parser.add_argument('--enable-short', action='store_true', help='Habilitar posiciones cortas')
    
    # Opciones para futuros
    run_parser.add_argument('--futures', action='store_true', help='Operar en el mercado de futuros en lugar de spot')
    run_parser.add_argument('--leverage', type=int, default=3, help='Apalancamiento para futuros (por defecto: 3x)')
    run_parser.add_argument('--hedge-mode', action='store_true', help='Usar modo hedge en futuros (permite posiciones largas y cortas simultáneas)')
    run_parser.add_argument('--trailing-stop', action='store_true', help='Usar trailing stop para futuros')
    run_parser.add_argument('--testnet', action='store_true', help='Usar testnet para futuros')
    
    # Comando: backtest (realizar backtesting)
    backtest_parser = subparsers.add_parser('backtest', help='Realizar backtesting de la estrategia')
    backtest_parser.add_argument('--symbol', default='ETHUSDT', help='Par de trading (por defecto: ETHUSDT)')
    backtest_parser.add_argument('--interval', default='1h', help='Intervalo de tiempo (por defecto: 1h)')
    backtest_parser.add_argument('--start-date', required=True, help='Fecha de inicio (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='Fecha de fin (YYYY-MM-DD, por defecto: hoy)')
    backtest_parser.add_argument('--capital', type=float, default=1000, help='Capital inicial (por defecto: 1000 USDT)')
    backtest_parser.add_argument('--enable-short', action='store_true', help='Habilitar posiciones cortas')
    
    # Opciones para futuros en backtest
    backtest_parser.add_argument('--futures', action='store_true', help='Backtest en el mercado de futuros')
    backtest_parser.add_argument('--leverage', type=int, default=3, help='Apalancamiento para futuros (por defecto: 3x)')
    
    # Comando: info (información sobre la cuenta y el mercado)
    info_parser = subparsers.add_parser('info', help='Mostrar información sobre la cuenta y el mercado')
    info_parser.add_argument('--balance', action='store_true', help='Mostrar balance de la cuenta')
    info_parser.add_argument('--positions', action='store_true', help='Mostrar posiciones abiertas')
    info_parser.add_argument('--futures', action='store_true', help='Mostrar información de futuros')
    info_parser.add_argument('--symbol', default='ETHUSDT', help='Par de trading (por defecto: ETHUSDT)')
    info_parser.add_argument('--testnet', action='store_true', help='Usar testnet para futuros')
    
    return parser.parse_args()


def map_interval(interval_str):
    """Mapea el intervalo de string a formato de Binance Client"""
    interval_map = {
        '1m': '1m',
        '3m': '3m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '8h': '8h',
        '12h': '12h',
        '1d': '1d',
        '3d': '3d',
        '1w': '1w',
        '1M': '1M'
    }
    return interval_map.get(interval_str, '1h')


def run_backtest(args):
    """Ejecuta el backtesting de la estrategia"""
    logger = setup_logging()
    logger.info(f"Iniciando backtesting para {args.symbol} desde {args.start_date}")
    
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Verificar API keys
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("Error: Las claves API no están configuradas en el archivo .env")
            sys.exit(1)
        
        # Configurar parámetros
        params = {
            'ENABLE_SHORT': args.enable_short,
        }
        
        if args.futures:
            params['LEVERAGE'] = args.leverage
            logger.info(f"Modo: FUTUROS, Apalancamiento: {args.leverage}x")
            
            # Importar BinanceFuturesBot y ejecutar su backtest (si está implementado)
            from binance_futures_bot import BinanceFuturesBot
            bot = BinanceFuturesBot(
                symbol=args.symbol,
                interval=map_interval(args.interval),
                params=params,
                testnet=True  # Usar testnet para backtest de futuros
            )
            # Nota: Aquí deberías implementar un método de backtest en la clase BinanceFuturesBot
            logger.error("Backtest para futuros aún no implementado")
            sys.exit(1)
        else:
            logger.info("Modo: SPOT")
            
            # Importar BinanceTradingBot y ejecutar su método de backtest
            from binance_bot import BinanceTradingBot
            bot = BinanceTradingBot(
                symbol=args.symbol,
                interval=map_interval(args.interval),
                params=params
            )
            
            # Ejecutar backtest
            result = bot.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.capital
            )
            
            if result:
                logger.info(f"Backtest completado con éxito. Resultados guardados.")
                
                # Generar informe
                stats = result['stats']
                logger.info(f"Estadísticas del backtest:")
                logger.info(f"Total operaciones: {stats['total_trades']}")
                logger.info(f"Win rate: {stats['win_rate']:.2f}%")
                logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
                logger.info(f"Drawdown máximo: {stats['max_drawdown']:.2f}%")
                logger.info(f"Informe guardado en: {os.path.abspath('backtest_report.txt')}")
            else:
                logger.error("Error al ejecutar el backtest")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error durante el backtest: {e}")
        sys.exit(1)


def show_account_info(args):
    """Muestra información de la cuenta y el mercado"""
    logger = setup_logging()
    
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Verificar API keys
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("Error: Las claves API no están configuradas en el archivo .env")
            sys.exit(1)
        
        if args.futures:
            # Importar e inicializar BinanceFuturesBot
            from binance_futures_bot import BinanceFuturesBot
            bot = BinanceFuturesBot(
                symbol=args.symbol,
                testnet=args.testnet
            )
            
            print("\n===== INFORMACIÓN DE CUENTA DE FUTUROS =====")
            
            if args.balance:
                account = bot.futures_client.futures_account()
                print(f"Balance total (USDT): {float(account['totalWalletBalance']):.2f}")
                print(f"Ganancias no realizadas: {float(account['totalUnrealizedProfit']):.2f}")
                print(f"Margen disponible: {float(account['availableBalance']):.2f}")
            
            if args.positions:
                positions = bot.get_position_info()
                print("\nPosiciones abiertas:")
                
                if positions['LONG']:
                    pos = positions['LONG']
                    print(f"LONG {args.symbol}: {float(pos['positionAmt'])} contratos")
                    print(f"Precio de entrada: {float(pos['entryPrice'])}")
                    print(f"PnL: {float(pos['unrealizedProfit']):.2f} USDT")
                    print(f"Precio de liquidación: {float(pos['liquidationPrice'])}")
                
                if positions['SHORT']:
                    pos = positions['SHORT']
                    print(f"SHORT {args.symbol}: {float(pos['positionAmt'])} contratos")
                    print(f"Precio de entrada: {float(pos['entryPrice'])}")
                    print(f"PnL: {float(pos['unrealizedProfit']):.2f} USDT")
                    print(f"Precio de liquidación: {float(pos['liquidationPrice'])}")
                
                if not positions['LONG'] and not positions['SHORT']:
                    print("No hay posiciones abiertas")
            
            # Información general si no se especificó nada concreto
            if not (args.balance or args.positions):
                funding_info = bot.get_funding_rate_history(limit=1)
                if funding_info is not None and not funding_info.empty:
                    print(f"\nTasa de financiamiento actual: {funding_info['fundingRate'].iloc[0]:.6f}%")
                
                ticker = bot.futures_client.futures_ticker(symbol=args.symbol)
                print(f"\nPrecio actual de {args.symbol}: {float(ticker['lastPrice'])}")
                print(f"Cambio 24h: {float(ticker['priceChangePercent'])}%")
        
        else:
            # Importar e inicializar BinanceTradingBot (spot)
            from binance_bot import BinanceTradingBot
            bot = BinanceTradingBot(
                symbol=args.symbol
            )
            
            print("\n===== INFORMACIÓN DE CUENTA SPOT =====")
            
            if args.balance:
                account = bot.client.get_account()
                print("Balances:")
                for asset in account['balances']:
                    if float(asset['free']) > 0 or float(asset['locked']) > 0:
                        print(f"{asset['asset']}: Libre = {asset['free']}, Bloqueado = {asset['locked']}")
            
            # Información general si no se especificó nada concreto
            if not args.balance:
                ticker = bot.client.get_symbol_ticker(symbol=args.symbol)
                print(f"\nPrecio actual de {args.symbol}: {float(ticker['price'])}")
                
                # Mostrar libro de órdenes
                depth = bot.client.get_order_book(symbol=args.symbol, limit=5)
                print("\nLibro de órdenes (5 mejores niveles):")
                
                print("Ofertas de compra:")
                for bid in depth['bids']:
                    print(f"Precio: {bid[0]}, Cantidad: {bid[1]}")
                
                print("\nOfertas de venta:")
                for ask in depth['asks']:
                    print(f"Precio: {ask[0]}, Cantidad: {ask[1]}")
    
    except Exception as e:
        logger.error(f"Error al mostrar información: {e}")
        sys.exit(1)


def main():
    """Función principal"""
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar API keys
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    
    logger = setup_logging()
    
    if not api_key or not api_secret:
        logger.error("Error: Las claves API no están configuradas en el archivo .env")
        logger.error("Por favor, crea un archivo .env con BINANCE_API_KEY y BINANCE_API_SECRET")
        sys.exit(1)
    
    # Analizar argumentos
    args = parse_args()
    
    # Ejecutar comando correspondiente
    if args.command == 'run':
        bot = RobustTradingBot(args)
        success = bot.start()
        if not success:
            sys.exit(1)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'info':
        show_account_info(args)
    else:
        logger.error("Error: Debe especificar un comando válido (run, backtest, info)")
        sys.exit(1)


if __name__ == "__main__":
    main()
