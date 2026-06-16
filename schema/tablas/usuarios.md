---
tabla: usuarios
descripcion: Clientes registrados en la plataforma. Estado activo/suspendido/baja (soft delete).
---

# usuarios

Clientes de la tienda. **Soft delete**: cuando un usuario se da de baja se cambia
`estado` a `'baja'` en lugar de borrarlo, para preservar integridad con sus
pedidos históricos.

## Columnas

| Columna         | Tipo        | Nullable | Default      | Descripción                                |
|-----------------|-------------|----------|--------------|--------------------------------------------|
| id              | bigint      | no       | serial       | PK.                                        |
| email           | text        | no       |              | Único.                                     |
| nombre          | text        | no       |              | Nombre para mostrar.                       |
| fecha_registro  | timestamptz | no       | now()        | Fecha de alta.                             |
| estado          | text        | no       | 'activo'     | `activo` \| `suspendido` \| `baja`.        |

## Relaciones

- (sin foreign keys salientes)
- Referenciada por: `direcciones.usuario_id`, `pedidos.usuario_id`.

## Notas de uso

- **Para listados o conteos comerciales**: filtra `WHERE estado = 'activo'`.
- **Para análisis histórico de pedidos**: no filtres por estado; un usuario en
  `baja` puede haber tenido pedidos válidos en el pasado.
- No hay tabla de autenticación: este modelo es solo de perfil de cliente.

## Ejemplos

```sql
-- Usuarios activos registrados en los últimos 30 días
SELECT id, nombre, email, fecha_registro
FROM usuarios
WHERE estado = 'activo' AND fecha_registro >= now() - interval '30 days'
ORDER BY fecha_registro DESC;

-- Distribución por estado
SELECT estado, COUNT(*) FROM usuarios GROUP BY estado;
```
