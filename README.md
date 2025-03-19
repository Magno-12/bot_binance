# Bot de Trading de Criptomonedas para Binance

Este proyecto implementa un bot de trading automatizado para la plataforma Binance, con soporte para análisis técnico, backtesting, generación de informes y optimización de parámetros.

## Características

- **Trading en Tiempo Real**: Opera automáticamente en Binance basándose en señales técnicas
- **Soporte para Posiciones Largas y Cortas**: Flexibilidad para aprovechar mercados alcistas y bajistas
- **Backtesting**: Prueba la estrategia con datos históricos antes de arriesgar capital real
- **Análisis de Rendimiento**: Seguimiento detallado y generación de informes
- **Gestión Dinámica de Riesgo**: Ajusta el tamaño de las posiciones según la volatilidad del mercado
- **Estructura Modular**: Código bien organizado siguiendo buenas prácticas de programación

## Estrategia Implementada

La estrategia de trading implementada se basa en los siguientes indicadores técnicos:

- **Bandas de Bollinger**: Para identificar condiciones de sobrecompra/sobreventa
- **RSI (Índice de Fuerza Relativa)**: Para confirmar las condiciones de sobrecompra/sobreventa
- **MACD (Convergencia/Divergencia de Medias Móviles)**: Para identificar cambios de tendencia
- **SMA y EMA**: Para analizar tendencias a medio plazo

## Requisitos

- Python 3.7+
- Cuenta en Binance (normal o testnet)
- Claves API de Binance

### Dependencias

```
python-binance>=1.0.16
pandas>=1.3.0
numpy>=1.20.0
python-dotenv>=0.19.0
ta>=0.10.0
matplotlib>=3.4.0
tabulate>=0.8.9
```

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tuusuario/binance-trading-bot.git
   cd binance-trading-bot
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura tus claves API de Binance:
   Crea un archivo `.env` en el directorio raíz con el siguiente contenido:
   ```
   BINANCE_API_KEY=tu_api_key
   BINANCE_API_SECRET=tu_api_secret
   TEST_MODE=True  # Cambia a False para operar con dinero real
   ```

## Uso

### Ejecutar el Bot en Tiempo Real

```bash
python main.py run --symbol ETHUSDT --interval 1h
```

Opciones:
- `--symbol`: Par de trading (por defecto: ETHUSDT)
- `--interval`: Intervalo de tiempo (por defecto: 1h)
- `--enable-short`: Habilitar posiciones cortas

### Realizar Backtesting

```bash
python main.py backtest --symbol ETHUSDT --interval 1h --start-date 2023-01-01 --end-date 2023-12-31
```

Opciones:
- `--symbol`: Par de trading
- `--interval`: Intervalo de tiempo
- `--start-date`: Fecha de inicio (YYYY-MM-DD)
- `--end-date`: Fecha de fin (opcional, por defecto: hoy)
- `--capital`: Capital inicial (por defecto: 1000 USDT)
- `--enable-short`: Habilitar posiciones cortas

### Generar Informe de Rendimiento

```bash
python main.py report --format html
```

Opciones:
- `--format`: Formato del informe (`text` o `html`)
- `--data-file`: Archivo de datos (por defecto: performance_data.json)

### Optimizar Parámetros (En desarrollo)

```bash
python main.py optimize --symbol ETHUSDT --start-date 2023-01-01
```

## Estructura del Proyecto

```
├── binance_bot.py         # Clase principal del bot
├── main.py                # Punto de entrada principal
├── performance_report.py  # Generador de informes
├── .env                   # Archivo de configuración (no incluido en el repositorio)
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

## Advertencia de Riesgo

El trading de criptomonedas conlleva un alto riesgo de pérdida. Este bot es una herramienta experimental y no garantiza beneficios. Úsalo bajo tu propia responsabilidad y nunca inviertas dinero que no puedas permitirte perder.

## Personalización

Puedes personalizar los parámetros de la estrategia modificando el diccionario `params` en la clase `BinanceTradingBot`:

```python
self.params = {
    'LENGTH_BB': 30,        # Período de las Bandas de Bollinger
    'MULT_BB': 2.0,         # Multiplicador de las Bandas de Bollinger
    'LENGTH_RSI': 14,       # Período del RSI
    'LENGTH_MACD_SHORT': 12, # Período corto del MACD
    'LENGTH_MACD_LONG': 26,  # Período largo del MACD
    'LENGTH_MACD_SIGNAL': 9, # Período de señal del MACD
    'STOP_LOSS_PERCENT': 2,  # Porcentaje de Stop Loss
    'TAKE_PROFIT_PERCENT': 4 # Porcentaje de Take Profit
}
```