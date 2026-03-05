-- lists all Glam rock bands ranked by longevity
SELECT
    band_name,
    (IFNULL(CAST(split AS SIGNED), 2020) - CAST(formed AS SIGNED)) AS lifespan
FROM metal_bands
WHERE main_style = 'Glam rock'
ORDER BY lifespan DESC;
