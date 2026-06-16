# Guía: cómo agregar tools al agente

Este documento explica cómo extender el agente DSPy de este proyecto con nuevas herramientas (tools), desde un cálculo determinístico en Python hasta una integración con una API externa con credenciales.

Tiene dos partes:

- **[Parte 1: guía para humanos](#parte-1-guía-para-humanos)** — autosuficiente. Conceptos, decisiones, paso a paso completo y dos ejemplos desarrollados y probados. Puedes agregar una tool de inicio a fin leyendo solo esta parte, sin usar el skill ni la Parte 2.
- **[Parte 2: recetario para agentes](#parte-2-recetario-para-agentes-fuente-de-verdad-del-skill-agregar-tool)** — complementa la Parte 1 con convenciones exactas, plantillas parametrizadas y comandos mecánicos. Es la fuente de verdad del skill `agregar-tool` de Claude Code (`.claude/skills/agregar-tool/`). No renombrar su encabezado: el skill lo usa como ancla.

---

## Parte 1: guía para humanos

### 1.1 ¿Qué es una tool?

Una tool es una **función Python normal** decorada con `@register_tool("nombre")` que vive en su propio archivo dentro de `tools/`. Cuando las tools están habilitadas, el agente usa `dspy.ReAct`: en cada turno el LLM lee los **docstrings** de las tools disponibles y decide si invoca alguna, con qué argumentos, y qué hacer con el resultado.

Reglas de oro:

- **1 archivo = 1 tool.** El archivo `tools/calculator.py` contiene la tool `calculator`.
- **El docstring es el prompt.** Es lo único que el LLM sabe de tu tool: qué hace, cuándo usarla, qué recibe y qué devuelve.
- **El LLM escribe los argumentos.** Trata todo input como no confiable (ver [1.8 Seguridad](#18-seguridad)).

### 1.2 Ciclo de vida: del YAML al agente

```
configs/config.yaml                core/wrapper.py                       tools/
┌─────────────────────┐   lee    ┌──────────────────────────┐   importa  ┌──────────────────────┐
│ tools:              │ ───────► │ DSPyWrapper._load_tools()│ ─────────► │ tools/<nombre>.py    │
│   enabled: true     │          │ importlib.import_module( │            │  @register_tool(...) │
│   available:        │          │   f"tools.{name}")       │            │  def <nombre>(...)   │
│     - <nombre>      │          └────────────┬─────────────┘            └──────────┬───────────┘
└─────────────────────┘                       │                                     │ el import ejecuta
                                              │ get_tools(names)                    │ el decorador
                                              ▼                                     ▼
                                  ┌──────────────────────────┐   consulta  ┌──────────────────────┐
                                  │ dspy.ReAct(sig,          │ ◄────────── │ tools/registry.py    │
                                  │   tools=[...],           │             │ _TOOL_REGISTRY       │
                                  │   max_iters=N)           │             │ {nombre: función}    │
                                  └──────────────────────────┘             └──────────────────────┘
```

En palabras: `DSPyWrapper.__init__` lee `tools.available` del YAML y, por cada nombre, hace `importlib.import_module(f"tools.{nombre}")` (`core/wrapper.py:73-84`). Ese import ejecuta el decorador `@register_tool`, que anota la función en el dict `_TOOL_REGISTRY` (`tools/registry.py`). Después `get_tools(names)` devuelve las funciones y se construye `dspy.ReAct` con ellas (`core/wrapper.py:94`). Si no hay tools habilitadas, el wrapper usa `dspy.Predict` (sin herramientas).

De aquí salen dos consecuencias importantes:

1. **Convención cuádruple** (obligatoria): nombre del módulo == nombre de la función == nombre registrado == entrada en el YAML. Si `tools/mi_tool.py` registra `@register_tool("otra_cosa")`, el import dinámico funciona pero `get_tools(["mi_tool"])` no la encuentra.
2. **Los errores de carga son silenciosos**: un typo en el YAML produce solo un warning en stderr (`tools/registry.py:36`) y un `ModuleNotFoundError` se ignora (`core/wrapper.py:80-81`). El agente arranca sin tu tool y nadie te avisa. Por eso el paso de verificación (1.9) no es opcional.

**Nunca necesitas tocar `core/wrapper.py` ni `tools/registry.py`** para agregar una tool.

### 1.3 El contrato de una tool

Toda tool cumple cuatro condiciones:

1. **Firma tipada simple.** Argumentos `str`, `int`, `float` o `bool` con nombres descriptivos. DSPy usa los type hints para indicarle al LLM qué enviar. Evita dicts, listas u objetos.
2. **Devuelve siempre `str`.** El retorno es la "observación" que el LLM lee. Si el resultado es tabular, formatea como tabla Markdown (como hace `supabase_query`).
3. **Nunca propaga excepciones.** Atrapa todo y devuelve un mensaje de error descriptivo en español. ReAct se lo muestra al LLM, que puede corregir y reintentar (por ejemplo, reformular un SQL inválido). Una excepción sin atrapar rompe el turno completo del agente.
4. **El docstring guía al LLM.** Estructura recomendada: qué hace → cuándo usarla → qué significa cada argumento → qué devuelve. Si depende de otras tools, dilo (así `supabase_query` indica llamar antes a `schema_list` y `schema_describe`).

Docstring **malo** (no dice cuándo usarla ni qué formato espera):

```python
@register_tool("exchange_rate")
def exchange_rate(d: str) -> str:
    """Obtiene el tipo de cambio."""
```

Docstring **bueno**:

```python
@register_tool("exchange_rate")
def exchange_rate(fecha: str) -> str:
    """Devuelve el tipo de cambio FIX peso/dólar publicado por Banxico
    para una fecha dada. Úsala cuando el usuario pregunte por el valor
    del dólar o necesites convertir montos entre MXN y USD.

    Args:
        fecha: Fecha en formato AAAA-MM-DD, por ejemplo "2026-06-09".
            Usa la tool datetime_now si necesitas la fecha de hoy.
    """
```

### 1.4 Árbol de decisión: ¿qué necesita tu tool?

Responde estas cinco preguntas antes de escribir código:

| # | Pregunta | Si la respuesta es sí... |
|---|----------|--------------------------|
| 1 | ¿Tiene parámetros que alguien querrá ajustar sin tocar código (límites, rutas, top_k)? | Sección nueva en `configs/config.yaml` + dataclass y loader en `core/config.py`. |
| 2 | ¿Necesita un secreto (API key, token, DSN)? | Variable en `.env` (el valor) y en `.env.example` (solo el nombre). **Jamás en el YAML ni en el código.** |
| 3 | ¿Usa un recurso costoso de crear (conexión a BD, cliente HTTP con sesión, índice vectorial)? | Adapter en `adapters/<dominio>/` (ABC + implementación) con lazy-init en la tool. |
| 4 | ¿Produce resultados ricos que la UI debería renderizar (tablas, SQL, gráficas)? | Buffer module-level `_LAST_*` + render opcional en `scripts/streamlit_app.py`. Nota: la sección "Razonamiento y herramientas" de la UI ya muestra los pasos de **cualquier** tool automáticamente; el render extra es solo para presentación rica. |
| 5 | ¿Requiere una librería nueva? | Declararla en `Pipfile` e instalarla en `.venv` con `uv pip install`. |

Una tool determinística simple responde "no" a todo: solo necesita su archivo y una línea en el YAML.

### 1.5 Taxonomía de tools (T1–T5)

| Tipo | Patrón | Ejemplo vivo en el repo |
|------|--------|------------------------|
| **T1** | Determinística pura, sin estado ni config | `tools/datetime_now.py` |
| **T2** | Con config en YAML y/o catálogo en disco | `tools/schema_list.py`, `tools/schema_describe.py` |
| **T3** | API externa con secreto en `.env` | (sin ejemplo en código; ver [1.11](#111-ejemplo-completo-b-weather_now-t3-api-externa--env)) |
| **T4** | Recurso costoso detrás de un adapter (lazy-init) | `tools/rag_search.py` + `adapters/vector/chroma.py` |
| **T5** | Con side-effects para la UI (buffer `_LAST_*`) | `tools/supabase_query.py` + `scripts/streamlit_app.py` |

Los tipos se combinan: `supabase_query` es a la vez T2 (sección `supabase` del YAML), T4 (pool de conexiones en `adapters/database/supabase.py`) y T5 (buffer `_LAST_QUERIES`). Clasifica por el rasgo dominante y suma lo que aplique.

### 1.6 Paso a paso genérico

Para una tool nueva llamada `<nombre>`:

**Paso 1 — Crea `tools/<nombre>.py`** con esta estructura mínima:

```python
"""Tool: una línea que describe qué hace."""

from __future__ import annotations

from tools.registry import register_tool


@register_tool("<nombre>")
def <nombre>(argumento: str) -> str:
    """Docstring para el LLM: qué hace, cuándo usarla, qué significa
    cada argumento y qué devuelve. En español.

    Args:
        argumento: Qué es y un ejemplo de valor válido.
    """
    argumento = (argumento or "").strip()
    if not argumento:
        return "Mensaje claro pidiendo el argumento, con un ejemplo."

    try:
        resultado = ...  # tu lógica
    except Exception as exc:
        return f"Error al <hacer la operación>: {exc}"

    return str(resultado)
```

**Paso 2 — Regístrala en `configs/config.yaml`**, dentro de `tools.available`:

```yaml
tools:
  enabled: true
  max_iters: 5
  available:
    - datetime_now
    - rag_search
    - supabase_query
    - schema_list
    - schema_describe
    - <nombre>        # ← tu tool
```

**Paso 3 — Condicionales** (solo lo que aplique según el árbol de 1.4):

- *Config tunable*: agrega la sección al YAML y el dataclass + loader en `core/config.py` (copia el patrón de `SupabaseConfig` / `load_supabase_config()`; el default del dataclass siempre `enabled: bool = False`).
- *Secreto*: agrega `MI_VARIABLE=` (sin valor) a `.env.example`, y la línea con el valor real a tu `.env` local. En el código, léela con `os.environ.get("MI_VARIABLE", "")` **dentro** de la función, no a nivel de módulo — así el módulo importa y se registra aunque la variable falte, y la tool puede devolver un mensaje útil.
- *Adapter*: crea `adapters/<dominio>/base.py` (ABC) y la implementación; en la tool usa el patrón lazy-init de `tools/rag_search.py` (`_adapter` module-level + `_get_adapter()`).
- *UI*: ver plantilla P5 en la Parte 2 y el patrón real en `tools/supabase_query.py`.
- *Dependencia nueva*: agrégala a `Pipfile` e instálala:

  ```bash
  VIRTUAL_ENV=.venv uv pip install <paquete>
  ```

**Paso 4 — Prueba** (los tres niveles de 1.9).

**Paso 5 — Documenta**: bullet en la sección "### 4. Tools (`tools/`)" de `CLAUDE.md`; si agregaste una variable de entorno, menciónala en "## Configuration". Si el agente necesita un *protocolo* de uso (orden entre tools, obligación de citar, prohibiciones), agrégalo a `agent.description` en el YAML; si solo necesita saber que la tool existe, el docstring basta.

### 1.7 Ideas de tools

Doce ideas categorizadas, de menor a mayor complejidad. Las marcadas con ★ están completamente desarrolladas más abajo.

**T1 — determinísticas puras:**

1. **`calculator`** ★ — evalúa expresiones aritméticas con `ast` (nunca `eval`). El LLM es malo para aritmética exacta; esta tool lo arregla.
2. **`unit_convert`** — km↔mi, °C↔°F, kg↔lb. Tabla de factores en el módulo.
3. **`text_stats`** — palabras, caracteres, frases de un texto dado.
4. **`curp_validate`** — validación estructural de una CURP (regex + dígito verificador). Útil en contexto mexicano.

**T2 — config/catálogo en disco:**

5. **`project_glossary`** — busca términos en un glosario de archivos `.md` con frontmatter, directorio configurable. Clon directo del patrón `schema_list`/`schema_describe`.
6. **`faq_lookup`** — responde desde un directorio de preguntas frecuentes en disco.

**T3 — API externa + secreto en `.env`:**

7. **`weather_now`** ★ — clima actual vía OpenWeatherMap (`OPENWEATHER_API_KEY`).
8. **`exchange_rate`** — tipo de cambio FIX vía la API SIE de Banxico (`BANXICO_TOKEN`, va en el header `Bmx-Token`).
9. **`web_search`** — búsqueda web vía Tavily o Brave (`TAVILY_API_KEY` / `BRAVE_API_KEY`).

**T4 — adapter con recurso costoso:**

10. **`http_fetch`** — descarga una URL y devuelve el texto truncado; adapter HTTP reutilizable con timeout y límite de bytes.
11. **`mongo_query`** — segundo motor de datos implementando `adapters/database/base.py`, como hizo Supabase.

**T5 — side-effects para la UI:**

12. **`chart_data`** — la tool devuelve un resumen textual al LLM, pero deja las filas en un buffer `_LAST_CHARTS` que la UI grafica con `st.bar_chart`.

### 1.8 Seguridad

- **El input de la tool lo escribe el LLM**, y el LLM lo construye a partir de lo que diga el usuario (o un documento RAG, o una fila de la BD). Trátalo como input hostil: valida formato, longitud y contenido. **Nunca** lo pases a `eval()`, `exec()`, un shell, o un string de SQL concatenado.
- **Secretos solo en `.env`**, leídos con `os.environ`. Nunca en el YAML (se versiona), nunca hardcodeados, nunca impresos en mensajes de error ni en el retorno de la tool.
- **Acota el output.** Todo lo que la tool devuelve entra al contexto del LLM y a la memoria de conversación. Referencias en el repo: `supabase_query` aplica un `LIMIT` implícito de 200 filas y trunca celdas largas. Para texto libre, corta a unos pocos KB con un sufijo `"... (truncado)"`.
- **Timeout en todo I/O.** Red: `requests.get(..., timeout=8)`. Base de datos: `statement_timeout_ms` en la sección `supabase` del YAML. Sin timeout, una API colgada congela el turno completo del agente.
- **Actitud de mínimo privilegio.** `SupabaseAdapter` usa tres capas (rol Postgres de solo lectura + transacción `BEGIN READ ONLY` + validación de tokens con sqlparse). Aplica el mismo criterio: si tu tool solo necesita leer, que no pueda escribir ni por accidente.

### 1.9 Cómo probar

No hay tests automatizados en este proyecto (decisión registrada en `CLAUDE.md`); la verificación es manual en tres niveles. Todos los comandos se ejecutan desde la raíz del proyecto.

**Nivel 1 — ¿Se registra?**

```bash
PYTHONPATH=. .venv/bin/python -c "
import importlib
importlib.import_module('tools.<nombre>')
from tools.registry import list_available
print(list_available())"
```

Debe aparecer `<nombre>` en la lista.

**Nivel 2 — ¿Funciona la función?** Llamada directa, probando también el camino de error:

```bash
PYTHONPATH=. .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from tools.<nombre> import <nombre>
print(<nombre>('<argumento de prueba>'))
print(<nombre>(''))  # camino de error: debe devolver un mensaje, no lanzar excepción"
```

> ⚠️ **Trampa de dotenv**: `load_dotenv()` solo se ejecuta en los entry-points (`scripts/streamlit_app.py`, `scripts/generate_schema_docs.py`). Un `python -c` suelto **no** carga `.env`. Si tu tool usa secretos, incluye `from dotenv import load_dotenv; load_dotenv('.env')` como en el ejemplo, o la tool reportará que no está configurada aunque tu `.env` esté correcto.

**Nivel 3 — ¿La carga el agente?** Este detecta los typos que el wrapper silencia (ver 1.2):

```bash
PYTHONPATH=. .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from core.wrapper import DSPyWrapper
print(DSPyWrapper().get_tool_names())"
```

Debe listar tu tool junto a las existentes. Después, prueba end-to-end:

```bash
PYTHONPATH=. streamlit run scripts/streamlit_app.py
```

Haz una pregunta que fuerce el uso de la tool y revisa la sección **"Razonamiento y herramientas"** bajo la respuesta: ahí se ve cada paso ReAct (pensamiento, tool invocada, argumentos y observación) sin necesidad de integración extra.

### 1.10 Ejemplo completo A: `calculator` (T1, determinística)

Una calculadora segura. La tentación es `eval(expresion)` — sería una vulnerabilidad de ejecución remota de código, porque el input viene del LLM (sección 1.8). En su lugar se parsea la expresión con `ast` y se evalúa con una **whitelist** de nodos: todo lo que no esté en la lista se rechaza, incluyendo nombres, llamadas a funciones y atributos (lo que cierra `__import__`, builtins y dunders).

El código completo de `tools/calculator.py` (probado: ver los casos al final):

```python
"""Tool: calculadora aritmética segura (evalúa con ast, nunca con eval)."""

from __future__ import annotations

import ast

from tools.registry import register_tool

_MAX_LONGITUD = 200
_MAX_EXPONENTE = 100

_OPERADORES_BINARIOS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_OPERADORES_UNARIOS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def _evaluar_nodo(nodo: ast.AST) -> float:
    """Evalúa recursivamente el AST aceptando SOLO aritmética pura (whitelist)."""
    if isinstance(nodo, ast.Expression):
        return _evaluar_nodo(nodo.body)
    if isinstance(nodo, ast.Constant):
        # bool es subclase de int: se rechaza explícitamente (True + 1 no es aritmética)
        if isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
            return nodo.value
        raise ValueError(f"constante no permitida: {nodo.value!r}")
    if isinstance(nodo, ast.BinOp):
        operador = _OPERADORES_BINARIOS.get(type(nodo.op))
        if operador is None:
            raise ValueError(f"operador no permitido: {type(nodo.op).__name__}")
        izquierda = _evaluar_nodo(nodo.left)
        derecha = _evaluar_nodo(nodo.right)
        # anti-DoS: sin esto, 9**9**9**9 intentaría construir un entero gigantesco
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > _MAX_EXPONENTE:
            raise ValueError(f"exponente fuera de rango (máximo {_MAX_EXPONENTE})")
        return operador(izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp):
        operador = _OPERADORES_UNARIOS.get(type(nodo.op))
        if operador is None:
            raise ValueError(f"operador unario no permitido: {type(nodo.op).__name__}")
        return operador(_evaluar_nodo(nodo.operand))
    # ast.Name, ast.Call, ast.Attribute y cualquier otro nodo caen aquí:
    # cierra __import__, builtins, atributos dunder, comprehensions, etc.
    raise ValueError(f"expresión no permitida: {type(nodo).__name__}")


@register_tool("calculator")
def calculator(expresion: str) -> str:
    """Evalúa una expresión aritmética y devuelve el resultado exacto.

    Acepta números, paréntesis y los operadores + - * / // % **.
    No acepta variables, funciones ni texto. Úsala cuando el usuario pida
    un cálculo o cuando necesites aritmética exacta en un paso intermedio
    (porcentajes, totales, conversiones).

    Args:
        expresion: Expresión aritmética, por ejemplo "(1234*5678) % 97".
    """
    expresion = (expresion or "").strip()
    if not expresion:
        return "Debes indicar una expresión aritmética, por ejemplo: 2 + 2 * 10."
    if len(expresion) > _MAX_LONGITUD:
        return f"La expresión es demasiado larga (máximo {_MAX_LONGITUD} caracteres)."

    try:
        arbol = ast.parse(expresion, mode="eval")
        resultado = _evaluar_nodo(arbol)
    except ZeroDivisionError:
        return "Error: división entre cero."
    except SyntaxError:
        return (
            "Error: la expresión no es aritmética válida. "
            "Solo se aceptan números, paréntesis y los operadores + - * / // % **."
        )
    except ValueError as exc:
        return f"Error: {exc}."
    except Exception as exc:  # red de seguridad: una tool nunca propaga excepciones
        return f"Error al evaluar la expresión: {exc}"

    if isinstance(resultado, float) and resultado.is_integer():
        resultado = int(resultado)
    return f"{expresion} = {resultado}"
```

Decisiones de seguridad, en orden de importancia:

| Defensa | Qué evita |
|---------|-----------|
| Whitelist de nodos AST (no blacklist) | Cualquier construcción no prevista se rechaza por defecto. `__import__('os')` muere como `Call`; `x` muere como `Name`. |
| `len(expresion) <= 200` | Expresiones gigantes que saturan el parser o el contexto. |
| Exponente de `Pow` acotado (`abs <= 100`) | `9**9**9**9` intentaría materializar un entero de ~10⁸ dígitos y colgaría el proceso. Se rechaza **antes** de computar. |
| `bool` rechazado en `Constant` | `True + 1` no es aritmética; en Python `bool` es subclase de `int` y pasaría sin el chequeo explícito. |
| Excepciones → string en español | El agente recibe la observación y puede reformular en lugar de morir. |

Registro en el YAML (`tools.available`):

```yaml
    - calculator
```

Verificación (resultados reales de la corrida de prueba):

```text
calculator("(1234*5678) % 97")   → "(1234*5678) % 97 = 51"
calculator("1 / 0")              → "Error: división entre cero."
calculator("__import__('os')")   → "Error: expresión no permitida: Call."
calculator("9**9**9**9")         → "Error: exponente fuera de rango (máximo 100)."
calculator("hola mundo")         → "Error: la expresión no es aritmética válida. ..."
```

Pregunta de prueba en Streamlit: *"¿Cuánto es (1234×5678) mod 97?"* — el agente debe invocar `calculator` y responder 51.

### 1.11 Ejemplo completo B: `weather_now` (T3, API externa + `.env`)

Integración con OpenWeatherMap. Lo que este ejemplo enseña aplica a cualquier API con credenciales: dónde vive el secreto, cómo se lee, cómo se manejan los errores HTTP de forma diferenciada y cómo se acota el output.

El código completo de `tools/weather_now.py`:

```python
"""Tool: clima actual de una ciudad vía OpenWeatherMap (requiere OPENWEATHER_API_KEY)."""

from __future__ import annotations

import os

import requests

from tools.registry import register_tool

_API_URL = "https://api.openweathermap.org/data/2.5/weather"
_TIMEOUT_S = 8


@register_tool("weather_now")
def weather_now(ciudad: str) -> str:
    """Devuelve el clima actual de una ciudad: temperatura, sensación térmica,
    condición y humedad. Úsala cuando el usuario pregunte por el clima, la
    temperatura o las condiciones meteorológicas de un lugar.

    Args:
        ciudad: Nombre de la ciudad, por ejemplo "Querétaro" o "Querétaro,MX".
    """
    ciudad = (ciudad or "").strip()
    if not ciudad:
        return "Debes indicar una ciudad, por ejemplo: 'Querétaro,MX'."

    # La key se lee DENTRO de la función: el módulo importa (y se registra)
    # aunque la variable no exista todavía.
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        return (
            "La tool de clima no está configurada: define OPENWEATHER_API_KEY "
            "en el archivo .env (ver .env.example)."
        )

    try:
        resp = requests.get(
            _API_URL,
            params={"q": ciudad, "appid": api_key, "units": "metric", "lang": "es"},
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return f"No se pudo consultar el servicio de clima (error de red): {exc}"

    if resp.status_code == 401:
        return (
            "La API key de OpenWeatherMap es inválida o está inactiva "
            "(revisa OPENWEATHER_API_KEY en .env)."
        )
    if resp.status_code == 404:
        return (
            f"No se encontró la ciudad '{ciudad}'. Prueba el formato 'Ciudad,CC' "
            "con código de país, por ejemplo 'Querétaro,MX'."
        )
    if resp.status_code != 200:
        return f"El servicio de clima respondió con error HTTP {resp.status_code}."

    try:
        data = resp.json()
        descripcion = data["weather"][0]["description"]
        temperatura = data["main"]["temp"]
        sensacion = data["main"]["feels_like"]
        humedad = data["main"]["humidity"]
        nombre = data.get("name") or ciudad
    except (KeyError, IndexError, ValueError) as exc:
        return f"Respuesta inesperada del servicio de clima: {exc}"

    # Output acotado: 4 líneas legibles, nunca el JSON crudo completo.
    return (
        f"Clima actual en {nombre}:\n"
        f"- Temperatura: {temperatura:.1f} °C (sensación térmica {sensacion:.1f} °C)\n"
        f"- Condición: {descripcion}\n"
        f"- Humedad: {humedad}%"
    )
```

Puntos clave del patrón T3:

- **Key lazy, dentro de la función.** Si se leyera a nivel de módulo y faltara, podrías tentarte a lanzar en el import — y el import dinámico del wrapper lo silenciaría (1.2): la tool simplemente desaparecería. Leyéndola adentro, la tool existe siempre y explica qué falta.
- **El camino "sin key" es un retorno legible y probado** — puedes verificar la tool completa sin tener cuenta de OpenWeatherMap (resultado real: `"La tool de clima no está configurada: define OPENWEATHER_API_KEY en el archivo .env (ver .env.example)."`).
- **Errores HTTP diferenciados**: 401 es un problema de configuración (lo arregla el humano), 404 es un problema del argumento (lo puede arreglar el LLM reformulando), y el resto se reporta tal cual. Darle al LLM la pista correcta es lo que hace que ReAct se recupere solo.
- **JSON crudo jamás**: la respuesta completa de la API trae decenas de campos; al contexto del LLM solo entran las 4 líneas útiles.

Checklist de integración (además del archivo):

1. `configs/config.yaml` → `- weather_now` en `tools.available`.
2. `.env.example` → línea `OPENWEATHER_API_KEY=` (sin valor).
3. `.env` → `OPENWEATHER_API_KEY=<tu_key_real>` (el humano la pega; jamás se versiona).
4. `Pipfile` → declarar `requests` (hoy está disponible solo como dependencia transitiva de streamlit; si se usa directamente, se declara).
5. `CLAUDE.md` → bullet en "### 4. Tools (`tools/`)" y mención de la variable en "## Configuration".

Pregunta de prueba en Streamlit: *"¿Cómo está el clima en Querétaro?"*.

---

## Parte 2: recetario para agentes (fuente de verdad del skill agregar-tool)

Esta parte asume los conceptos de la Parte 1 y los convierte en reglas mecánicas. Está pensada para que un agente (Claude Code vía el skill `agregar-tool`, u otro) ejecute sin ambigüedad. **Si algo aquí contradice al código real del repo, el código gana**: usa las tools existentes como referencia viva y reporta el drift.

### 2.1 Invariantes de naming

| Elemento | Regla | Ejemplo |
|----------|-------|---------|
| Convención cuádruple | módulo == función == nombre registrado == entrada YAML | `tools/calculator.py` / `def calculator` / `@register_tool("calculator")` / `- calculator` |
| Nombres de tools | `snake_case`, descriptivo, en inglés o español consistente | `weather_now`, `schema_list` |
| Variables de entorno | `MAYÚSCULAS_CON_GUIÓN_BAJO`, sufijo según tipo: `_API_KEY`, `_TOKEN`, `_URL` | `OPENWEATHER_API_KEY`, `BANXICO_TOKEN`, `SUPABASE_DB_URL` |
| Sección YAML de config | minúsculas, igual al dominio de la tool | `supabase:`, `vectorizer:` |
| Dataclass de config | `<Nombre>Config` en `core/config.py`, campos con defaults, **siempre** `enabled: bool = False` | `SupabaseConfig` |
| Loader de config | `load_<nombre>_config(config_path: Path \| None = None)` | `load_supabase_config()` |
| Buffer de UI | `_LAST_<COSA>` module-level + `get_last_<cosa>()` + `clear_last_<cosa>()` + `_record_<cosa>()` | `_LAST_QUERIES` en `tools/supabase_query.py` |
| Idioma | Docstrings, comentarios y mensajes de error en español; identificadores en el idioma del módulo de referencia | — |

### 2.2 Tabla archivo-por-archivo según tipo

✔ = obligatorio · ◐ = solo si aplica · — = no tocar

| Archivo | T1 puro | T2 config | T3 API+secreto | T4 adapter | T5 UI |
|---------|:-:|:-:|:-:|:-:|:-:|
| `tools/<nombre>.py` (crear) | ✔ | ✔ | ✔ | ✔ | ✔ |
| `configs/config.yaml` → `tools.available` | ✔ | ✔ | ✔ | ✔ | ✔ |
| `configs/config.yaml` → sección nueva | — | ✔ | ◐ | ✔ | — |
| `core/config.py` → dataclass + loader | — | ✔ | ◐ | ✔ | — |
| `.env.example` (nombre) + `.env` (valor, lo pone el humano) | — | — | ✔ | ◐ (si hay DSN) | — |
| `adapters/<dominio>/base.py` + implementación (crear) | — | — | — | ✔ | — |
| `scripts/streamlit_app.py` (clear/get/render) | — | — | — | — | ◐ |
| `Pipfile` + `uv pip install` | ◐ si hay dependencia nueva (cualquier tipo) | | | | |
| `CLAUDE.md` | ✔ siempre: bullet en "### 4. Tools (`tools/`)"; env var en "## Configuration"; dep nueva en "## Commands" | | | | |
| `config_examples/*.yaml` | ◐ si se agregó una sección nueva al YAML (mantener templates en sync) | | | | |
| `agent.description` (YAML) | ◐ solo si el agente necesita protocolo de uso (criterio en 2.5) | | | | |
| `core/wrapper.py`, `tools/registry.py` | — **nunca** se tocan para agregar una tool | | | | |

### 2.3 Plantillas

Marcadores a sustituir: `<NOMBRE>` (snake_case), `<DESCRIPCION_LLM>` (docstring completo), `<ENV_VAR>`, `<DOMINIO>`, `<Nombre>` (CamelCase).

#### P1 — Tool determinística pura (T1)

```python
"""Tool: <una línea sobre qué hace>."""

from __future__ import annotations

from tools.registry import register_tool


@register_tool("<NOMBRE>")
def <NOMBRE>(argumento: str) -> str:
    """<DESCRIPCION_LLM>

    Args:
        argumento: <qué es y ejemplo de valor válido>.
    """
    argumento = (argumento or "").strip()
    if not argumento:
        return "<mensaje pidiendo el argumento, con ejemplo>"

    try:
        resultado = ...  # lógica determinística; validar input antes de usarlo
    except Exception as exc:
        return f"Error al <operación>: {exc}"

    return str(resultado)
```

#### P2 — Tool con config en YAML (T2)

`tools/<NOMBRE>.py`:

```python
"""Tool: <una línea>."""

from __future__ import annotations

from tools.registry import register_tool


@register_tool("<NOMBRE>")
def <NOMBRE>(argumento: str) -> str:
    """<DESCRIPCION_LLM>"""
    from core.config import load_<NOMBRE>_config

    config = load_<NOMBRE>_config()
    if not config.enabled:
        return "La tool <NOMBRE> no está habilitada en la configuración."

    # usar config.<campo>...
    return "..."
```

Agregar a `core/config.py` (junto a los demás dataclasses):

```python
@dataclass
class <Nombre>Config:
    """Configuración de la tool <NOMBRE>."""

    enabled: bool = False
    # campos tunables con default seguro, p. ej.:
    # max_resultados: int = 10


def load_<NOMBRE>_config(config_path: Path | None = None) -> <Nombre>Config:
    """Carga la sección <NOMBRE> del YAML."""
    path = config_path or _DEFAULT_CONFIG_PATH
    data = _load_yaml(path)
    seccion = data.get("<NOMBRE>") or {}
    return <Nombre>Config(
        enabled=bool(seccion.get("enabled", False)),
        # max_resultados=int(seccion.get("max_resultados", 10)),
    )
```

Agregar a `configs/config.yaml`:

```yaml
<NOMBRE>:
  enabled: true
  # max_resultados: 10
```

#### P3 — Tool de API externa con secreto (T3)

Espejo del ejemplo `weather_now` de la Parte 1 (sección 1.11) — usar ese código como plantilla, sustituyendo URL, `<ENV_VAR>`, parámetros y parseo de respuesta. Reglas no negociables del patrón:

- `os.environ.get("<ENV_VAR>", "")` **dentro** de la función; si falta → retorno explicativo que mencione `.env` y `.env.example`.
- `timeout=` SIEMPRE en la llamada HTTP (8 s por defecto).
- Errores diferenciados: credencial inválida (humano) vs argumento malo (el LLM puede reintentar) vs error genérico.
- Output acotado y formateado; jamás el JSON crudo.
- Agregar `<ENV_VAR>=` a `.env.example` y al `.env` local (línea vacía; el valor lo pone el humano).

#### P4 — Tool con adapter (T4)

`adapters/<DOMINIO>/base.py`:

```python
"""Interfaz base para adaptadores de <DOMINIO>."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Base<Nombre>Adapter(ABC):
    """Protocolo para adaptadores de <DOMINIO>."""

    @abstractmethod
    def operacion(self, argumento: str) -> dict:
        """Contrato de la operación principal; documentar el dict de retorno."""
        ...
```

`adapters/<DOMINIO>/<implementacion>.py`: clase concreta que hereda de la ABC, recibe el `<Nombre>Config` en `__init__` y crea el recurso costoso (pool, cliente, índice) de forma lazy. Referencia viva: `adapters/database/supabase.py` (pool + timeouts + validación) o `adapters/vector/chroma.py` (cliente persistente).

`tools/<NOMBRE>.py` con lazy-init (patrón exacto de `tools/rag_search.py`):

```python
"""Tool: <una línea>."""

from __future__ import annotations

from tools.registry import register_tool

_adapter = None
_config = None


def _get_adapter():
    """Inicializa el adaptador de forma lazy (una sola vez por proceso)."""
    global _adapter, _config
    if _adapter is None:
        from adapters.<DOMINIO>.<implementacion> import <Nombre>Adapter
        from core.config import load_<NOMBRE>_config

        _config = load_<NOMBRE>_config()
        if not _config.enabled:
            return None, _config
        _adapter = <Nombre>Adapter(_config)
    return _adapter, _config


@register_tool("<NOMBRE>")
def <NOMBRE>(argumento: str) -> str:
    """<DESCRIPCION_LLM>"""
    adapter, config = _get_adapter()
    if adapter is None:
        return "La tool <NOMBRE> no está habilitada en la configuración."

    try:
        resultado = adapter.operacion(argumento)
    except Exception as exc:
        return f"Error al <operación>: {exc}"

    return str(resultado)  # formatear acotado (tabla Markdown si es tabular)
```

(P4 requiere también el dataclass + loader de P2.)

#### P5 — Tool con side-effects para la UI (T5)

En `tools/<NOMBRE>.py`, además de la lógica (P1–P4 según corresponda):

```python
_LAST_<COSAS>: list[dict] = []  # buffer module-level que la UI lee


def get_last_<cosas>() -> list[dict]:
    """Devuelve una copia de las operaciones de la ronda actual (para la UI)."""
    return list(_LAST_<COSAS>)


def clear_last_<cosas>() -> None:
    """Vacía el buffer (la UI lo llama antes de cada respond())."""
    _LAST_<COSAS>.clear()


def _record_<cosa>(datos: dict, error: str | None = None) -> None:
    """Anota la operación en el buffer para que la UI la muestre."""
    _LAST_<COSAS>.append({**datos, "error": error})
```

Dentro de la tool: llamar `_record_<cosa>(...)` en éxito **y** en cada camino de error (la UI debe poder mostrar qué falló). Referencia viva: `tools/supabase_query.py` (`_record_query`).

En `scripts/streamlit_app.py`, tres toques (patrón exacto de las líneas ~168-180 donde ya se hace con `supabase_query`):

```python
# 1) Antes de wrapper.respond(): importar y limpiar
try:
    from tools.<NOMBRE> import clear_last_<cosas>, get_last_<cosas>
    clear_last_<cosas>()
except ImportError:
    get_last_<cosas> = lambda: []

# 2) Después de respond(): capturar en session_state
st.session_state.last_<cosas> = get_last_<cosas>()

# 3) Render: función _render_<cosas>(items) con st.expander por item,
#    st.code/st.dataframe/st.bar_chart según el contenido, st.error si item["error"].
#    Llamarla junto a _render_db_queries() y _render_trajectory().
```

Recordatorio: la trayectoria ReAct (`_render_trajectory`) ya muestra cualquier tool sin este paso; P5 es solo para presentación rica (dataframes, gráficas, SQL resaltado).

### 2.4 Comandos de verificación exactos

Ejecutar desde la raíz del proyecto, en este orden. Si alguno falla, corregir antes de continuar.

```bash
# 1. Registro: el módulo importa y se registra con el nombre correcto
PYTHONPATH=. .venv/bin/python -c "
import importlib
importlib.import_module('tools.<NOMBRE>')
from tools.registry import list_available
print(list_available())"

# 2. Función: camino feliz Y camino de error (nunca debe lanzar excepción)
PYTHONPATH=. .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from tools.<NOMBRE> import <NOMBRE>
print(<NOMBRE>('<argumento válido>'))
print(<NOMBRE>(''))"

# 3. Carga vía wrapper: detecta typos YAML↔módulo que el wrapper silencia
PYTHONPATH=. .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from core.wrapper import DSPyWrapper
print(DSPyWrapper().get_tool_names())"

# 4. End-to-end: preguntar algo que fuerce la tool y revisar
#    "Razonamiento y herramientas" bajo la respuesta
PYTHONPATH=. streamlit run scripts/streamlit_app.py
```

Notas:

- El comando 2 **debe** incluir `load_dotenv` si la tool usa secretos (trampa documentada en 1.9).
- El comando 3 no requiere una API key válida: construye el wrapper completo pero no hace llamadas al LLM (`dspy.configure` y `dspy.ReAct` no contactan la API). Las keys solo hacen falta para el end-to-end del comando 4.
- Si la tool nueva no aparece en el comando 3 pero sí en el 1: el nombre en `tools.available` del YAML no coincide con el nombre registrado (convención cuádruple rota).

### 2.5 CLAUDE.md y `agent.description`

**CLAUDE.md** (siempre):

- Bullet del módulo nuevo en la sección "### 4. Tools (`tools/`)" (una línea: nombre + qué hace + particularidades como buffers de UI).
- Si hay env var nueva: mencionarla en "## Configuration" junto a las existentes.
- Si hay dependencia nueva: agregarla al comando de instalación en "## Commands".

**`agent.description`** en `configs/config.yaml` (solo si se cumple el criterio):

- ✔ Agregar instrucciones cuando el agente necesita un **protocolo de uso**: orden obligatorio entre tools (como "llama `schema_list` antes de escribir SQL"), obligación de citar fuentes, prohibiciones, o desambiguación entre tools parecidas.
- ✘ No agregar nada si el agente solo necesita saber que la tool existe — para eso basta el docstring. Inflar `agent.description` degrada todas las respuestas.

### 2.6 Checklist de reporte final

Al terminar, reportar al usuario:

1. **Archivos creados** y **modificados** (lista con rutas).
2. **Pendientes manuales**: típicamente "pega el valor real de `<ENV_VAR>` en `.env`" (el agente nunca pide ni escribe el valor del secreto).
3. **Verificación ejecutada**: salida de los comandos 1–3 de la sección 2.4 (y si se corrió el 4).
4. **Pregunta de prueba** sugerida para validar en Streamlit (una que fuerce el uso de la tool).
5. **Recordatorio de no commitear** si el flujo del proyecto requiere validación previa del usuario.

---

*Verificado contra el código del repo el 2026-06-09. Si encuentras una discrepancia entre esta guía y el código, el código gana — y agradece un fix a la guía.*
