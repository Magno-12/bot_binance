# Bot de Trading de Criptomonedas para Binance

Este proyecto implementa un bot de trading automatizado para la plataforma Binance, con soporte para trading en los mercados spot y futuros. El bot incluye características avanzadas como análisis técnico, backtesting, generación de informes, gestión de riesgo dinámica y optimización de parámetros.

## Características Principales

### Características Generales
- **Trading en Tiempo Real**: Opera automáticamente en Binance basándose en señales técnicas
- **Soporte para Posiciones Largas y Cortas**: Flexibilidad para aprovechar mercados alcistas y bajistas
- **Backtesting**: Prueba la estrategia con datos históricos antes de arriesgar capital real
- **Análisis de Rendimiento**: Seguimiento detallado y generación de informes
- **Gestión Dinámica de Riesgo**: Ajusta el tamaño de las posiciones según la volatilidad del mercado
- **Estructura Modular**: Código bien organizado siguiendo buenas prácticas de programación
- **Sistema de Notificaciones**: Recibe alertas por correo electrónico sobre operaciones y errores
- **Endpoint de Salud**: Monitoreo del estado del bot a través de una API HTTP
- **Persistencia de Datos**: Guardado automático de estado y copia de seguridad 
- **Manejo de Errores Robusto**: Recuperación automática ante fallos

### Características Mercado Spot
- **Trading Convencional**: Compra y venta de criptomonedas en el mercado spot
- **Gestión de Órdenes**: Colocación y seguimiento de órdenes de mercado y límite
- **Stop Loss y Take Profit**: Protección de capital y aseguramiento de ganancias

### Características Mercado Futuros (Nuevo)
- **Apalancamiento Configurable**: Opera con múltiplos de tu capital (hasta 125x)
- **Modo Hedge**: Posibilidad de tener posiciones largas y cortas simultáneas
- **Trailing Stop**: Stop Loss dinámico que se ajusta con el movimiento del mercado
- **Análisis de Financiamiento**: Monitoreo de tasas de financiamiento para optimizar entradas
- **Gestión de Margen**: Configuración automática del tipo de margen (aislado/cruzado)
- **Mayor Flexibilidad**: Facilidad para abrir posiciones cortas sin necesidad de préstamos

## Estrategia Implementada

La estrategia de trading implementada se basa en los siguientes indicadores técnicos:

- **Bandas de Bollinger**: Para identificar condiciones de sobrecompra/sobreventa
- **RSI (Índice de Fuerza Relativa)**: Para confirmar las condiciones de sobrecompra/sobreventa
- **MACD (Convergencia/Divergencia de Medias Móviles)**: Para identificar cambios de tendencia
- **SMA y EMA**: Para analizar tendencias a medio plazo
- **ATR (Average True Range)**: Para medir volatilidad y ajustar el tamaño de posiciones

### Lógica de Señales

El bot opera usando una combinación de señales para identificar oportunidades:

#### Señales de Compra (Long)
1. Precio cruza por encima de la banda inferior de Bollinger y RSI < 30
2. O bien, MACD cruza por encima de la línea de señal

#### Señales de Venta (Short)
1. Precio cruza por encima de la banda superior de Bollinger y RSI > 70
2. O bien, MACD cruza por debajo de la línea de señal

#### Gestión de Riesgo
- El tamaño de posición se ajusta dinámicamente según la volatilidad medida por ATR
- Se implementan stop loss y take profit para cada operación
- En futuros, se utiliza liquidación aislada para limitar pérdidas potenciales

## Requisitos

- Python 3.7+
- Cuenta en Binance (normal o testnet)
- Claves API de Binance con permisos de lectura y trading

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
   
   # Configuración de notificaciones por correo (opcional)
   ENABLE_EMAIL_NOTIFICATIONS=True
   EMAIL_SENDER=tu_correo@gmail.com
   EMAIL_PASSWORD=tu_contraseña_de_app
   EMAIL_RECIPIENT=destinatario@gmail.com
   EMAIL_SERVER=smtp.gmail.com
   EMAIL_PORT=587
   ```

## Guía de Uso

### Monitoreo del Estado

El bot incluye un servidor HTTP que permite monitorear su estado accediendo a `http://localhost:8080/health` (o el puerto especificado en la variable `PORT`).

### Trading en Tiempo Real en Mercado Spot

```bash
# Modo básico con configuración predeterminada
python main.py run --symbol ETHUSDT --interval 1h

# Habilitar posiciones cortas (requiere tener margen activado en Binance)
python main.py run --symbol BTCUSDT --interval 15m --enable-short
```

### Trading en Tiempo Real en Mercado Futuros (Nuevo)

```bash
# Modo básico de futuros con apalancamiento 3x
python main.py run --futures --symbol ETHUSDT --interval 1h

# Futuros con mayor apalancamiento y modo hedge
python main.py run --futures --symbol BTCUSDT --interval 15m --leverage 5 --hedge-mode

# Futuros con trailing stop para gestión dinámica de stop loss
python main.py run --futures --symbol ETHUSDT --interval 4h --trailing-stop

# Usar testnet para pruebas sin riesgo
python main.py run --futures --testnet --symbol BTCUSDT --interval 1h
```

### Backtesting

```bash
# Backtesting en mercado spot
python main.py backtest --symbol ETHUSDT --interval 1h --start-date 2023-01-01 --end-date 2023-12-31

# Backtesting con capital inicial personalizado
python main.py backtest --symbol BTCUSDT --interval 4h --start-date 2023-01-01 --capital 5000

# Backtesting con posiciones cortas habilitadas
python main.py backtest --symbol ETHUSDT --interval 1d --start-date 2023-01-01 --enable-short

# Backtesting en mercado futuros (simulación)
python main.py backtest --futures --leverage 3 --symbol BTCUSDT --interval 1h --start-date 2023-01-01
```

### Obtener Información de Cuenta y Mercado

```bash
# Información general de mercado spot
python main.py info --symbol ETHUSDT

# Ver balance de cuenta en spot
python main.py info --balance

# Información de mercado de futuros
python main.py info --futures --symbol BTCUSDT

# Ver posiciones abiertas en futuros
python main.py info --futures --positions
```

## Estructura del Proyecto

```
├── binance_bot.py         # Clase principal del bot para mercado spot
├── binance_futures_bot.py # Clase principal del bot para mercado futuros
├── main.py                # Punto de entrada principal con manejo robusto de errores
├── performance_report.py  # Generador de informes de rendimiento
├── notification_system.py # Sistema de notificaciones por correo
├── data_persistence.py    # Sistema de persistencia y backup de datos
├── health_endpoint.py     # Servidor HTTP para monitoreo
├── logs/                  # Directorio de logs (creado automáticamente)
├── data/                  # Directorio de datos (creado automáticamente)
├── backups/               # Directorio de copias de seguridad (creado automáticamente)
├── reports/               # Directorio de informes (creado automáticamente)
├── .env                   # Archivo de configuración (no incluido en el repositorio)
├── requirements.txt       # Dependencias del proyecto
├── Dockerfile             # Archivo para construir imagen Docker
├── render.yaml            # Configuración para despliegue en Render
└── README.md              # Este archivo
```

## Componentes Principales

### 1. BinanceTradingBot (binance_bot.py)
Este componente maneja el trading en el mercado spot de Binance. Sus funciones principales incluyen:
- Obtener datos históricos de precios
- Calcular indicadores técnicos
- Generar señales de compra/venta
- Ejecutar órdenes en el mercado
- Gestionar posiciones abiertas
- Backtesting de estrategias

### 2. BinanceFuturesBot (binance_futures_bot.py)
Este nuevo componente maneja el trading en el mercado de futuros de Binance. Sus funciones principales incluyen:
- Configuración de apalancamiento y modo de posición
- Obtención de datos específicos de futuros (funding rates, etc.)
- Colocación de órdenes con características especiales (trailing stop, etc.)
- Gestión de posiciones largas y cortas (incluido modo hedge)
- Monitoreo de precios de liquidación

### 3. RobustTradingBot (main.py)
Este componente proporciona una capa de robustez y manejo de errores para la ejecución:
- Gestión de ciclo de vida del bot
- Recuperación automática ante fallos
- Reintentos con backoff exponencial
- Manejo de señales del sistema operativo
- Guardado periódico del estado
- Generación de informes diarios

### 4. Sistemas Auxiliares
- **NotificationSystem**: Envía alertas y notificaciones por correo electrónico
- **DataPersistence**: Gestiona el guardado de datos y copias de seguridad
- **HealthCheckServer**: Ofrece un endpoint HTTP para monitoreo externo
- **PerformanceReportGenerator**: Genera informes detallados de rendimiento

## Personalización

### Parámetros de Estrategia

Puedes personalizar los parámetros de la estrategia modificando el diccionario `params` en las clases `BinanceTradingBot` o `BinanceFuturesBot`:

```python
# Parámetros para mercado spot
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

# Parámetros adicionales para mercado futuros
self.params.update({
    'LEVERAGE': 3,           # Apalancamiento (3x)
    'HEDGE_MODE': False,     # Modo de cobertura
    'TRAILING_STOP': False,  # Activar stop loss dinámico
    'TRAILING_STOP_CALLBACK': 1.0, # Porcentaje de callback para trailing stop
})
```

### Implementación de Nuevas Estrategias

Para implementar una nueva estrategia, puedes modificar la clase `SignalGenerator` creando nuevos métodos para detectar señales de entrada/salida:

```python
class MyCustomSignalGenerator:
    @staticmethod
    def check_buy_signal(df, current_idx):
        # Tu lógica personalizada aquí
        return buy_condition
        
    @staticmethod
    def check_sell_signal(df, current_idx):
        # Tu lógica personalizada aquí
        return sell_condition
```

Luego asigna tu generador de señales en la configuración del bot:

```python
bot = BinanceTradingBot(symbol="BTCUSDT")
bot.signal_generator = MyCustomSignalGenerator
```

## Despliegue

### Docker

Puedes ejecutar el bot en un contenedor Docker:

```bash
# Construir la imagen
docker build -t crypto-trading-bot .

# Ejecutar el contenedor
docker run -d --name trading-bot --env-file .env crypto-trading-bot
```

### Render

El proyecto incluye un archivo `render.yaml` para despliegue automático en Render:

1. Crea una cuenta en Render.com
2. Conecta tu repositorio de GitHub
3. Configura las variables de entorno en Render
4. Despliega la aplicación

## Advertencia de Riesgo

El trading de criptomonedas conlleva un alto riesgo de pérdida, especialmente al utilizar apalancamiento en el mercado de futuros. Este bot es una herramienta experimental y no garantiza beneficios. Úsalo bajo tu propia responsabilidad y nunca inviertas dinero que no puedas permitirte perder.

Recomendaciones de seguridad:
- Comienza con pequeñas cantidades para probar
- Usa la testnet para desarrollo y pruebas
- Limita el acceso de la API a "solo trading" (sin permisos de retiro)
- Configura direcciones IP permitidas en tu API de Binance
- Monitorea regularmente el comportamiento del bot

## Solución de Problemas

### Problemas Comunes

1. **Error de conectividad con Binance**
   ```
   Problema: Error al obtener datos históricos
   Solución: Verifica tu conexión a internet y que las API keys sean correctas
   ```

2. **Insuficiente balance**
   ```
   Problema: La cantidad a comprar es demasiado pequeña
   Solución: Asegúrate de tener suficiente USDT en tu cuenta
   ```

3. **Error en cálculo de indicadores**
   ```
   Problema: No se pueden calcular indicadores técnicos
   Solución: Verifica que estás obteniendo suficientes datos históricos
   ```

4. **Error en ejecución de órdenes**
   ```
   Problema: Error al colocar orden
   Solución: Verifica los filtros de Binance (tamaño mínimo, precio, etc.)
   ```

### Logs

Los logs se guardan en el directorio `logs/` y pueden ayudar a diagnosticar problemas:

```bash
# Ver los últimos 100 logs
tail -n 100 logs/bot_$(date +%Y-%m-%d).log

# Buscar errores en los logs
grep ERROR logs/bot_*.log
```

## Desarrollo Futuro

Características planificadas para futuras versiones:

- Optimización automática de parámetros mediante algoritmos genéticos
- Interfaz web para monitoreo y control
- Soporte para múltiples pares de trading simultáneos
- Estrategias avanzadas (grid trading, DCA, etc.)
- Integración con chatbots (Telegram, Discord) para control remoto
- Sistema de machine learning para predecir movimientos de mercado

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Contacto y Soporte

Para preguntas, sugerencias o reportes de errores, por favor abre un issue en GitHub o contacta al desarrollador en [tu-email@ejemplo.com].

---

## Nota sobre el Mercado de Futuros

La implementación del mercado de futuros es una característica poderosa que permite operar con apalancamiento y abrir posiciones cortas con mayor facilidad. Sin embargo, estas capacidades vienen con riesgos adicionales:

1. **Liquidación**: Con apalancamiento, tu posición puede ser liquidada si el precio se mueve en tu contra
2. **Apalancamiento excesivo**: Un alto apalancamiento puede resultar en pérdidas rápidas y sustanciales
3. **Tasas de financiamiento**: En contratos perpetuos, hay pagos periódicos entre traders largos y cortos

Recomendamos comenzar con apalancamiento bajo (2-3x) y tamaños de posición pequeños mientras te familiarizas con este mercado.

### Ejemplo de Uso Seguro en Futuros

```bash
# Comenzar con testnet para practicar sin riesgo
python main.py run --futures --testnet --symbol ETHUSDT --interval 1h --leverage 2

# Usar modo hedge para proteger posiciones existentes
python main.py run --futures --symbol BTCUSDT --interval 4h --leverage 2 --hedge-mode

# Implementar trailing stop para proteger ganancias
python main.py run --futures --symbol ETHUSDT --interval 1h --trailing-stop
```