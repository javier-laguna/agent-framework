# Metodología de trabajo (Git)

Rama principal: **main**. Una sola rama de integración; todo lo estable se sube ahí.

## Commits

Mensajes con prefijo corto para saber qué cambió:

| Prefijo   | Uso |
|-----------|-----|
| `feat:`   | Nueva funcionalidad (ej. feat: adapter Gemini). |
| `fix:`    | Corrección de bug. |
| `docs:`   | Solo documentación (README, comentarios). |
| `config:` | Cambios en YAML, .gitignore, Pipfile. |
| `refactor:` | Reorganización de código sin cambiar comportamiento. |

Ejemplo: `feat: memoria de conversación con resumen por hechos`

## Ramas

- **main**: código estable. Solo se hace push de cambios ya probados.
- Si se quiere probar algo aislado: crear una rama corta (ej. `feature/nombre` o `fix/descripcion`), hacer commits ahí, luego merge a main y borrar la rama. No obligatorio si siempre trabajáis sobre main.

## Versiones

Opcional: marcar hitos con tags semánticos.

```bash
git tag -a v0.1.0 -m "Primera versión: wrapper, OpenAI/Gemini, Streamlit, memoria"
git push origin v0.1.0
```

## Resumen

1. Trabajar sobre main (o una rama corta si hace falta).
2. Commit a menudo con mensajes `prefijo: descripción`.
3. Push a `origin main` cuando el estado sea estable.
4. Tags cuando se cierre un hito (v0.1.0, v1.0.0, etc.).
