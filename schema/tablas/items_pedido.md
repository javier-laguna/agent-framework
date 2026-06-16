---
tabla: items_pedido
descripcion: Líneas de cada pedido (producto + cantidad + precio snapshot).
---

# items_pedido

Líneas que componen un pedido. Cada fila es una combinación
`(pedido_id, producto_id)` con su cantidad y el **precio congelado** al momento
de la compra. Si el producto cambia de precio después, los items previos
conservan el precio original.

## Columnas

| Columna                    | Tipo    | Nullable | Default | Descripción                                                   |
|----------------------------|---------|----------|---------|---------------------------------------------------------------|
| id                         | bigint  | no       | serial  | PK.                                                           |
| pedido_id                  | bigint  | no       |         | FK → `pedidos.id` (ON DELETE CASCADE).                        |
| producto_id                | bigint  | no       |         | FK → `productos.id`.                                          |
| cantidad                   | integer | no       |         | `> 0`.                                                        |
| precio_unitario_centavos   | integer | no       |         | Precio del producto al momento de la compra (snapshot). `>= 0`.|

## Relaciones

- `pedido_id` → `pedidos.id` (muchos-a-uno).
- `producto_id` → `productos.id` (muchos-a-uno).

## Notas de uso

- **Ingresos por producto/categoría**: siempre usa
  `cantidad * precio_unitario_centavos`, NO el precio actual de `productos`.
- **Para top productos**: agrega `SUM(cantidad)` agrupando por `producto_id`.
- No hay constraint UNIQUE sobre `(pedido_id, producto_id)` — un pedido podría
  tener dos líneas del mismo producto si la UI lo permite.

## Ejemplos

```sql
-- Top 10 productos por unidades vendidas (sólo pedidos efectivos)
SELECT p.nombre,
       SUM(ip.cantidad) AS unidades,
       SUM(ip.cantidad * ip.precio_unitario_centavos) / 100.0 AS ingresos_mxn
FROM items_pedido ip
JOIN pedidos pe ON pe.id = ip.pedido_id
JOIN productos p ON p.id = ip.producto_id
WHERE pe.estado IN ('pagado','enviado','entregado')
GROUP BY p.id, p.nombre
ORDER BY unidades DESC
LIMIT 10;

-- Tamaño promedio del carrito (items por pedido)
SELECT AVG(items_por_pedido)::numeric(10,2) AS items_promedio
FROM (
  SELECT pedido_id, COUNT(*) AS items_por_pedido
  FROM items_pedido GROUP BY pedido_id
) t;
```
