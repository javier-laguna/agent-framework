---
tabla: categorias
descripcion: Catálogo de categorías de productos. Cada producto pertenece a una.
---

# categorias

Tabla de referencia con las categorías comerciales del catálogo. Es pequeña y
estable: se crea una vez al configurar la tienda y rara vez cambia.

## Columnas

| Columna     | Tipo        | Nullable | Default | Descripción                       |
|-------------|-------------|----------|---------|-----------------------------------|
| id          | bigint      | no       | serial  | PK auto-incremental.              |
| nombre      | text        | no       |         | Nombre único de la categoría.     |
| descripcion | text        | sí       |         | Descripción libre.                |
| creada_en   | timestamptz | no       | now()   | Fecha de creación.                |

## Relaciones

- (sin foreign keys salientes)
- Referenciada por: `productos.categoria_id`.

## Notas de uso

- 10 categorías fijas: Electrónica, Hogar, Ropa, Libros, Deportes, Juguetes,
  Belleza, Alimentos, Mascotas, Oficina.
- Se asume que `nombre` se muestra en UI tal cual (no hay slug separado).

## Ejemplos

```sql
-- Listar las categorías con cuántos productos activos tienen
SELECT c.nombre, COUNT(p.id) FILTER (WHERE p.activo) AS productos_activos
FROM categorias c
LEFT JOIN productos p ON p.categoria_id = c.id
GROUP BY c.id, c.nombre
ORDER BY productos_activos DESC;
```
