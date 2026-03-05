-- lists all genres by their rating
SELECT t4.name, SUM(t2.rate) AS rating
FROM tv_show_ratings t2
INNER JOIN tv_shows t1 ON t2.show_id = t1.id
INNER JOIN tv_show_genres t3 ON t3.show_id = t1.id
INNER JOIN tv_genres t4 ON t3.genre_id = t4.id
GROUP BY t4.name
ORDER BY rating DESC
