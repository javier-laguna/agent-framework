---
tabla: direcciones
descripcion: Direcciones de envío de los usuarios. Un usuario puede tener varias; una es principal.
---

# direcciones

Cada usuario puede tener 1 o más direcciones. Se borran en cascada si se borra
físicamente el usuario (raro: el flujo normal es soft-delete vía
`usuarios.estado = 'baja'`).

## Columnas

| Columna        | Tipo    | Nullable | Default | Descripción                                          |
|----------------|---------|----------|---------|------------------------------------------------------|
| id             | bigint  | no       | serial  | PK.                                                  |
| usuario_id     | bigint  | no       |         | FK → `usuarios.id` (ON DELETE CASCADE).              |
| etiqueta       | text    | no       |         | Alias humano: `casa`, `trabajo`, etc.                |
| calle          | text    | no       |         | Calle y número.                                      |
| ciudad         | text    | no       |         | Ciudad.                                              |
| estado         | text    | no       |         | Entidad federativa (no confundir con `usuarios.estado`).|
| codigo_postal  | text    | no       |         | CP en formato libre (5 dígitos en seed).             |
| es_principal   | boolean | no       | false   | Hay como máximo una principal por usuario (no enforced en BD).|

## Relaciones

- `usuario_id` → `usuarios.id` (muchos-a-uno).
- Referenciada por: `pedidos.direccion_envio_id`.

## Notas de uso

- **Cuidado con la columna `estado`**: es la entidad federativa, NO el estado
  de baja como en `usuarios.estado`. Si necesitas el estado del cliente, hace
  JOIN con `usuarios` y usa `usuarios.estado`.
- Para obtener la dirección "preferida" de un usuario, usa
  `WHERE es_principal = true LIMIT 1` (la constraint no está enforced).

## Ejemplos

```sql
-- Direcciones principales por ciudad
SELECT ciudad, COUNT(*) AS direcciones_principales
FROM direcciones
WHERE es_principal = true
GROUP BY ciudad
ORDER BY direcciones_principales DESC;
```
