---
tabla: pedidos
descripcion: Pedidos realizados por los usuarios. Cada uno tiene 1+ items y 0+ pagos.
---

# pedidos

Cabecera de cada orden. El detalle (productos comprados) vive en `items_pedido`.
Las transacciones de pago, en `pagos`. Un pedido puede tener múltiples intentos
de pago (típicamente uno, dos si el primero falló).

## Columnas

| Columna              | Tipo        | Nullable | Default | Descripción                                                   |
|----------------------|-------------|----------|---------|---------------------------------------------------------------|
| id                   | bigint      | no       | serial  | PK.                                                           |
| usuario_id           | bigint      | no       |         | FK → `usuarios.id`. El cliente que hizo el pedido.            |
| direccion_envio_id   | bigint      | sí       |         | FK → `direcciones.id`. Nullable (pickup en tienda).           |
| estado               | text        | no       |         | `pendiente` \| `pagado` \| `enviado` \| `entregado` \| `cancelado`. |
| total_centavos       | integer     | no       |         | Total denormalizado (suma de items al momento de la compra).  |
| creado_en            | timestamptz | no       | now()   | Fecha de creación del pedido.                                 |
| actualizado_en       | timestamptz | no       | now()   | Última modificación (estado, envío, etc.).                    |

## Relaciones

- `usuario_id` → `usuarios.id` (muchos-a-uno).
- `direccion_envio_id` → `direcciones.id` (muchos-a-uno, opcional).
- Referenciada por: `items_pedido.pedido_id`, `pagos.pedido_id`.

## Notas de uso

- **Filtro común para análisis de ingresos**: `estado IN ('pagado','enviado','entregado')`.
  Los `pendiente` y `cancelado` no representan venta efectiva.
- **`total_centavos` está denormalizado**: refleja el total al momento del
  pedido. Para sumas históricas usa esto directamente, NO recalcules con
  `productos.precio_centavos`.
- **Filtros temporales**: `creado_en >= now() - interval 'X days'` para
  ventanas móviles.

## Ejemplos

```sql
-- Ingresos por mes en el último año
SELECT date_trunc('month', creado_en) AS mes,
       COUNT(*) AS pedidos,
       SUM(total_centavos) / 100.0 AS ingresos_mxn
FROM pedidos
WHERE estado IN ('pagado','enviado','entregado')
  AND creado_en >= now() - interval '1 year'
GROUP BY mes
ORDER BY mes;

-- Pedidos pendientes mayores a 5000 MXN (revisión manual)
SELECT id, usuario_id, total_centavos / 100.0 AS total_mxn, creado_en
FROM pedidos
WHERE estado = 'pendiente' AND total_centavos > 500000
ORDER BY creado_en;
```
