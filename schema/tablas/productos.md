---
tabla: productos
descripcion: Productos del catálogo. Pertenecen a una categoría y tienen stock, precio y estado activo.
---

# productos

Cada fila es un SKU del catálogo. Los productos inactivos (`activo = false`) no
se muestran en el catálogo público pero se conservan para mantener integridad
histórica con `items_pedido` (los pedidos pasados pueden seguir referenciándolos).

## Columnas

| Columna           | Tipo        | Nullable | Default | Descripción                                       |
|-------------------|-------------|----------|---------|---------------------------------------------------|
| id                | bigint      | no       | serial  | PK.                                               |
| sku               | text        | no       |         | Identificador comercial único legible.            |
| nombre            | text        | no       |         | Nombre comercial del producto.                    |
| descripcion       | text        | sí       |         | Descripción libre.                                |
| categoria_id      | bigint      | no       |         | FK → `categorias.id`.                             |
| precio_centavos   | integer     | no       |         | Precio en centavos (MXN). `>= 0`.                 |
| stock             | integer     | no       | 0       | Unidades disponibles. `>= 0`.                     |
| activo            | boolean     | no       | true    | Si false, no se vende al público.                 |
| creado_en         | timestamptz | no       | now()   | Fecha de alta.                                    |

## Relaciones

- `categoria_id` → `categorias.id` (muchos-a-uno).
- Referenciada por: `items_pedido.producto_id`.

## Notas de uso

- **Precio**: siempre en centavos. Para mostrar en pesos divide entre 100.0.
- **Snapshot de precio**: cuando se compra, el precio se copia a
  `items_pedido.precio_unitario_centavos`. No uses `productos.precio_centavos`
  para calcular ingresos históricos: usa el snapshot.
- **Filtrar activos**: `WHERE activo = true` para listados públicos.

## Ejemplos

```sql
-- Productos más caros por categoría
SELECT c.nombre AS categoria, p.nombre, p.precio_centavos / 100.0 AS precio_mxn
FROM productos p
JOIN categorias c ON c.id = p.categoria_id
WHERE p.activo
ORDER BY c.nombre, p.precio_centavos DESC;

-- Productos sin stock pero activos (riesgo de venta sin inventario)
SELECT id, sku, nombre, stock FROM productos
WHERE activo = true AND stock = 0;
```
