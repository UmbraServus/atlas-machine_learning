-- lists all shows containing at least one linked genre.
-- in asc order by title and id
SELECT t1.title, t2.genre_id
FROM tv_shows t1
INNER JOIN tv_show_genres t2
ON t2.show_id = t1.id
ORDER BY t1.title asc, t2.genre_id asc;
