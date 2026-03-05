-- lists all Glam rock bands ranked by longevity
SELECT band_name, (split, 2020) - formed AS lifespan
FROM metal_bands
WHERE main_style = 'Glam rock'
ORDER BY lifespan DESC;
