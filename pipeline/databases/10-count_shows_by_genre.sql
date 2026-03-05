-- count shows by genre
SELECT t1.name AS genre, COUNT(t2.show_id) AS number_of_shows
FROM tv_genres t1
INNER JOIN tv_show_genres t2
ON t1.id = t2.genre_id
GROUP BY t1.name
ORDER BY number_of_shows DESC;
