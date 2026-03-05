--  lists shows with no genre link
SELECT t1.title, t2.genre_id
FROM tv_shows t1
LEFT JOIN tv_show_genres t2
ON t2.show_id = t1.id
WHERE t2.genre_id IS NULL
ORDER BY t1.title asc, t2.genre_id asc;
