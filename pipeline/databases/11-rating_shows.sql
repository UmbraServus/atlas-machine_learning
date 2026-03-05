-- lists all shows by their rating
SELECT t1.title, SUM(t2.rate) AS rating
FROM tv_show_ratings t2
INNER JOIN tv_shows t1
ON t2.show_id = t1.id
GROUP BY t1.title
ORDER BY rating DESC
