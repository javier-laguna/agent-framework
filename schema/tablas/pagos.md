---
tabla: pagos
descripcion: Transacciones de pago. Un pedido puede tener varios intentos (reintentos tras fallido).
---

# pagos

Cada intento de pago es una fila. Lo normal es un pago `completado` por pedido,
pero hay casos con un `fallido` seguido de un `completado` (reintento), o un
`pendiente` que nunca prosperó.

## Columnas

| Columna         | Tipo        | Nullable | Default | Descripción                                                   |
|-----------------|-------------|----------|---------|---------------------------------------------------------------|
| id              | bigint      | no       | serial  | PK.                                                           |
| pedido_id       | bigint      | no       |         | FK → `pedidos.id` (ON DELETE CASCADE).                        |
| metodo          | text        | no       |         | `tarjeta` \| `transferencia` \| `efectivo`.                   |
| monto_centavos  | integer     | no       |         | Monto de este intento. `>= 0`.                                |
| estado          | text        | no       |         | `pendiente` \| `completado` \| `fallido` \| `reembolsado`.    |
| creado_en       | timestamptz | no       | now()   | Cuándo se registró este intento.                              |

## Relaciones

- `pedido_id` → `pedidos.id` (muchos-a-uno).

## Notas de uso

- **Un pedido = N pagos**. Para saber si un pedido está realmente cobrado,
  verifica que exista al menos un `pagos.estado = 'completado'` para ese
  `pedido_id`.
- **`monto_centavos` puede no coincidir con `pedidos.total_centavos`** en
  pagos parciales o reembolsos (no implementado en el seed actual, pero
  contemplado por la estructura).
- **Tasa de conversión / fallos**: cuenta pagos por `(metodo, estado)`.

## Ejemplos

```sql
-- Tasa de éxito por método de pago
SELECT metodo,
       COUNT(*) FILTER (WHERE estado = 'completado') AS completados,
       COUNT(*) FILTER (WHERE estado = 'fallido')    AS fallidos,
       ROUND(
         100.0 * COUNT(*) FILTER (WHERE estado = 'completado') / NULLIF(COUNT(*), 0),
         2
       ) AS pct_exito
FROM pagos
GROUP BY metodo
ORDER BY pct_exito DESC;

-- Pedidos con más de un intento de pago
SELECT pedido_id, COUNT(*) AS intentos,
       STRING_AGG(estado, ', ' ORDER BY creado_en) AS secuencia
FROM pagos
GROUP BY pedido_id
HAVING COUNT(*) > 1
ORDER BY intentos DESC;
```
