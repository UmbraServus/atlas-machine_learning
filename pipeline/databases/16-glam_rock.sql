-- lists all Glam rock bands ranked by longevity
SELECT band_name, (IFNULL(split, 2020) - formed) AS lifespan
FROM
    metal_bands
WHERE
    LOWER(style) LIKE '%glam rock%'
ORDER BY
    lifespan DESC;