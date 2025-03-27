# futures_main.py
import argparse
import logging
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from binance_futures_bot import BinanceFuturesBot

# Configuración de logging
def setup_logging():
    """Configura el sistema de logging con rotación de archivos"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Archivo de log con fecha
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = f"{log_dir}/futures_bot_{today}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("futures_main")

def parse_args():
    """Analiza los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Bot de Trading de Futuros para Binance')
    
    # Comando principal
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: run (ejecutar bot en tiempo real)
    run_parser = subparsers.add_parser('run', help='Ejecutar el bot en tiempo real')
    run_parser.add_argument('--symbol', default='ETHUSDT', help='Par de trading (por defecto: ETHUSDT)')
    run_parser.add_argument('--interval', default='1h', help='Intervalo de tiempo (por defecto: 1h)')
    run_parser.add_argument('--leverage', type=int, default=3, help='Apalancamiento (por defecto: 3x)')
    run_parser.add_argument('--hedge-mode', action='store_true', help='Habilitar modo de cobertura (hedge mode)')
    run_parser.add_argument('--trailing-stop', action='store_true', help='Habilitar trailing stop')
    run_parser.add_argument('--testnet', action='store_true', help='Usar testnet de Binance')
    
    # Comando: info (obtener información de cuenta y mercados)
    info_parser = subparsers.add_parser('info', help='Obtener información de cuenta y mercados')
    info_parser.add_argument('--balance', action='store_true', help='Mostrar balance de cuenta')
    info_parser.add_argument('--positions', action='store_true', help='Mostrar posiciones abiertas')
    info_parser.add_argument('--funding', action='store_true', help='Mostrar tasas de financiamiento')
    info_parser.add_argument('--symbol', default='ETHUSDT', help='Par de trading (por defecto: ETHUSDT)')
    info_parser.add_argument('--testnet', action='store_true', help='Usar testnet de Binance')
    
    # Comando: close (cerrar posiciones abiertas)
    close_parser = subparsers.add_parser('close', help='Cerrar posiciones abiertas')
    close_parser.add_argument('--symbol', default='ETHUSDT', help='Par de trading (por defecto: ETHUSDT)')
    close_parser.add_argument('--position-side', choices=['LONG', 'SHORT', 'ALL'], default='ALL', 
                             help='Lado de posición a cerrar (por defecto: ALL)')
    close_parser.add_argument('--testnet', action='store_true', help='Usar testnet de Binance')
    
    return parser.parse_args()

def run_bot(args):
    """Ejecuta el bot de futuros en tiempo real"""
    logger = setup_logging()
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar API keys
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("Error: Las claves API no están configuradas en el archivo .env")
        logger.error("Por favor, crea un archivo .env con BINANCE_API_KEY y BINANCE_API_SECRET")
        sys.exit(1)
    
    # Configurar parámetros
    params = {
        'LEVERAGE': args.leverage,
        'HEDGE_MODE': args.hedge_mode,
        'TRAILING_STOP': args.trailing_stop,
    }
    
    logger.info(f"Iniciando bot de futuros para {args.symbol} en intervalo {args.interval}")
    logger.info(f"Configuración: Apalancamiento {args.leverage}x, Hedge Mode: {args.hedge_mode}, "
               f"Trailing Stop: {args.trailing_stop}, Testnet: {args.testnet}")
    
    try:
        # Inicializar el bot
        bot = BinanceFuturesBot(
            symbol=args.symbol,
            interval=args.interval,
            params=params,
            testnet=args.testnet
        )
        
        # Ejecutar estrategia
        bot.run_strategy()
        
    except KeyboardInterrupt:
        logger.info("Bot detenido manualmente")
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        sys.exit(1)

def show_account_info(args):
    """Muestra información de la cuenta y mercados"""
    logger = setup_logging()
    
    # Cargar variables de entorno
    load_dotenv()
    
    try:
        # Inicializar el bot (solo para consultas)
        bot = BinanceFuturesBot(
            symbol=args.symbol,
            testnet=args.testnet
        )
        
        # Mostrar información según argumentos
        if args.balance:
            # Obtener balance
            account = bot.futures_client.futures_account()
            
            print("\n===== BALANCE DE CUENTA =====")
            print(f"Balance total (USDT): {float(account['totalWalletBalance']):.2f}")
            print(f"Balance no realizado (USDT): {float(account['totalUnrealizedProfit']):.2f}")
            print(f"Margen disponible (USDT): {float(account['availableBalance']):.2f}")
            print(f"Margen usado (USDT): {float(account['totalPositionInitialMargin']):.2f}")
            
            # Mostrar activos
            print("\nBalance por activo:")
            for asset in account['assets']:
                if float(asset['walletBalance']) > 0:
                    print(f"  {asset['asset']}: {float(asset['walletBalance']):.8f} "
                         f"(Disponible: {float(asset['availableBalance']):.8f})")
        
        if args.positions:
            # Obtener posiciones abiertas
            positions = bot.futures_client.futures_position_information()
            
            print("\n===== POSICIONES ABIERTAS =====")
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            if open_positions:
                for pos in open_positions:
                    symbol = pos['symbol']
                    side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
                    amount = abs(float(pos['positionAmt']))
                    entry_price = float(pos['entryPrice'])
                    mark_price = float(pos['markPrice'])
                    pnl = float(pos['unrealizedProfit'])
                    pnl_percent = ((mark_price / entry_price) - 1) * 100 * (1 if side == "LONG" else -1)
                    leverage = pos['leverage']
                    
                    print(f"Symbol: {symbol}")
                    print(f"Side: {side}")
                    print(f"Size: {amount} contratos")
                    print(f"Entry Price: {entry_price}")
                    print(f"Mark Price: {mark_price}")
                    print(f"PnL: {pnl:.4f} USDT ({pnl_percent:.2f}%)")
                    print(f"Leverage: {leverage}x")
                    print(f"Liquidation Price: {float(pos['liquidationPrice'])}")
                    print("----------------------------")
            else:
                print("No hay posiciones abiertas actualmente")
        
        if args.funding:
            # Obtener tasas de financiamiento
            funding_history = bot.get_funding_rate_history(limit=10)
            
            if funding_history is not None and not funding_history.empty:
                print("\n===== TASAS DE FINANCIAMIENTO RECIENTES =====")
                print(f"Symbol: {args.symbol}")
                
                for _, row in funding_history.iterrows():
                    time = row['fundingTime'].strftime('%Y-%m-%d %H:%M:%S')
                    rate = row['fundingRate']
                    print(f"Time: {time}, Rate: {rate:.6f}%")
                
                # Obtener próxima tasa
                try:
                    funding_info = bot.futures_client.futures_funding_rate(symbol=args.symbol)
                    next_funding_time = datetime.fromtimestamp(int(funding_info[0]['nextFundingTime']) / 1000)
                    print(f"\nPróximo financiamiento: {next_funding_time}")
                except Exception as e:
                    logger.error(f"Error al obtener próxima tasa de financiamiento: {e}")
            else:
                print(f"No se pudieron obtener tasas de financiamiento para {args.symbol}")
        
        # Si no se especificó ninguna opción, mostrar resumen
        if not (args.balance or args.positions or args.funding):
            # Mostrar información general
            exchange_info = bot.futures_client.futures_exchange_info()
            server_time = datetime.fromtimestamp(exchange_info['serverTime'] / 1000)
            
            print("\n===== INFORMACIÓN DEL MERCADO DE FUTUROS =====")
            print(f"Hora del servidor: {server_time}")
            
            # Mostrar información del símbolo especificado
            symbol_info = None
            for s in exchange_info['symbols']:
                if s['symbol'] == args.symbol:
                    symbol_info = s
                    break
            
            if symbol_info:
                print(f"\nInformación de {args.symbol}:")
                print(f"Tipo de contrato: {symbol_info['contractType']}")
                print(f"Status: {symbol_info['status']}")
                print(f"Precio base: {symbol_info['pricePrecision']} decimales")
                print(f"Cantidad mínima: {symbol_info.get('lotSizeFilter', {}).get('minQty', 'N/A')}")
                
                # Obtener tickers
                ticker = bot.futures_client.futures_ticker(symbol=args.symbol)
                if ticker:
                    print(f"Precio actual: {float(ticker['lastPrice'])}")
                    print(f"Cambio 24h: {float(ticker['priceChangePercent'])}%")
                    print(f"Volumen 24h: {float(ticker['volume'])}")
            else:
                print(f"No se encontró información para el símbolo {args.symbol}")
    
    except Exception as e:
        logger.error(f"Error al obtener información: {e}")
        sys.exit(1)

def close_positions(args):
    """Cierra las posiciones especificadas"""
    logger = setup_logging()
    
    # Cargar variables de entorno
    load_dotenv()
    
    try:
        # Inicializar el bot
        bot = BinanceFuturesBot(
            symbol=args.symbol,
            testnet=args.testnet
        )
        
        # Obtener posiciones abiertas
        positions = bot.get_position_info()
        
        if args.position_side == 'ALL':
            # Cerrar todas las posiciones
            if positions['LONG'] or positions['SHORT']:
                if positions['LONG']:
                    bot.close_position('LONG')
                if positions['SHORT']:
                    bot.close_position('SHORT')
                print(f"Todas las posiciones de {args.symbol} han sido cerradas")
            else:
                print(f"No hay posiciones abiertas para {args.symbol}")
        
        elif args.position_side == 'LONG':
            # Cerrar solo posiciones largas
            if positions['LONG']:
                bot.close_position('LONG')
                print(f"Posición LONG de {args.symbol} ha sido cerrada")
            else:
                print(f"No hay posición LONG abierta para {args.symbol}")
        
        elif args.position_side == 'SHORT':
            # Cerrar solo posiciones cortas
            if positions['SHORT']:
                bot.close_position('SHORT')
                print(f"Posición SHORT de {args.symbol} ha sido cerrada")
            else:
                print(f"No hay posición SHORT abierta para {args.symbol}")
    
    except Exception as e:
        logger.error(f"Error al cerrar posiciones: {e}")
        sys.exit(1)

def main():
    """Función principal"""
    args = parse_args()
    
    if args.command == 'run':
        run_bot(args)
    elif args.command == 'info':
        show_account_info(args)
    elif args.command == 'close':
        close_positions(args)
    else:
        print("Error: Debe especificar un comando válido (run, info, close)")
        sys.exit(1)

if __name__ == "__main__":
    main()