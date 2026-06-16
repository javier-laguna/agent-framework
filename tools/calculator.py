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
