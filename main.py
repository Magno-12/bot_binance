# robust_main.py
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
            
            # Intentar cargar parámetros guardados
            saved_params = self.persistence.load_data('bot_params.json')
            if saved_params:
                params.update(saved_params)
                self.logger.info(f"Parámetros cargados desde archivo persistente")
            
            # Inicializar el bot
            self.bot = BinanceTradingBot(
                symbol=self.args.symbol,
                interval=map_interval(self.args.interval),
                params=params
            )
            
            # Verificar conectividad
            if not self.check_api_connectivity():
                raise Exception("No se pudo establecer conexión con la API de Binance")
            
            self.logger.info(f"Bot inicializado para {self.args.symbol} en intervalo {self.args.interval}")
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
        
        # Inicializar bot
        if not self.initialize_bot():
            self.logger.error("No se pudo inicializar el bot. Abortando.")
            return False
        
        # Notificar inicio
        self.notifier.notify_restart("Inicio del sistema")
        
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
            return
        
        # Calcular indicadores
        df = self.bot.params['calculate_indicators'](df, self.bot.params)
        
        # Verificar si hay posiciones abiertas
        self.bot.check_position_status()
        
        # Si no hay posición abierta, buscar señales de entrada
        if not self.bot.in_position:
            buy_signal = self.bot.params['check_buy_signal'](df, -1)
            sell_signal = self.bot.params['check_sell_signal'](df, -1)
            
            if buy_signal:
                self.logger.info("Señal de compra detectada")
                self.bot.place_buy_order()
            elif sell_signal and self.bot.params.get('ENABLE_SHORT', False):
                self.logger.info("Señal de venta en corto detectada")
                self.bot.place_sell_short_order()
    
    def _save_bot_state(self):
        """Guarda el estado actual del bot"""
        try:
            # Guardar estado del rastreador de rendimiento
            if hasattr(self.bot, 'tracker') and hasattr(self.bot.tracker, 'trades'):
                data = {
                    'trades': self.bot.tracker.trades,
                    'equity_curve': self.bot.tracker.equity_curve
                }
                self.persistence.save_data(data, 'performance_data.json')
                
            # Guardar parámetros actuales
            self.persistence.save_data(self.bot.params, 'bot_params.json')
            
            # Guardar estado actual (posición, etc.)
            state = {
                'in_position': self.bot.in_position,
                'position_side': self.bot.position_side,
                'entry_price': self.bot.entry_price,
                'stop_loss': self.bot.stop_loss,
                'take_profit': self.bot.take_profit,
                'last_update': datetime.now().isoformat()
            }
            self.persistence.save_data(state, 'bot_state.json')
            
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
            
            # Enviar resumen
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'current_balance': current_balance
            }
            
            self.notifier.notify_daily_summary(stats)
            self.logger.info("Resumen diario enviado")
            
        except Exception as e:
            self.logger.error(f"Error al enviar resumen diario: {e}")
    
    def handle_error(self, error):
        """Maneja errores durante la ejecución del bot"""
        self.consecutive_errors += 1
        
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
                self.notifier.notify_restart("Reinicio tras errores")
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
            self.notifier.notify_error("Bot detenido", "El sistema ha sido detenido correctamente.")
            
            self.logger.info("Limpieza finalizada correctamente")
            
        except Exception as e:
            self.logger.error(f"Error durante la limpieza final: {e}")


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
    
    # Comando: backtest (realizar backtesting)
    backtest_parser = subparsers.add_parser('backtest', help='Realizar backtesting de la estrategia')
    backtest_parser.add_argument('--symbol', default='ETHUSDT', help='Par de trading (por defecto: ETHUSDT)')
    backtest_parser.add_argument('--interval', default='1h', help='Intervalo de tiempo (por defecto: 1h)')
    backtest_parser.add_argument('--start-date', required=True, help='Fecha de inicio (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='Fecha de fin (YYYY-MM-DD, por defecto: hoy)')
    backtest_parser.add_argument('--capital', type=float, default=1000, help='Capital inicial (por defecto: 1000 USDT)')
    backtest_parser.add_argument('--enable-short', action='store_true', help='Habilitar posiciones cortas')
    
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
        # Para backtest, usamos la implementación original
        from main import run_backtest
        run_backtest(args)
    else:
        logger.error("Error: Debe especificar un comando válido (run, backtest)")
        sys.exit(1)


if __name__ == "__main__":
    main()
