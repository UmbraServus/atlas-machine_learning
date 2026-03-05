-- selects city, value and orders them by desc average of value.
SELECT city, AVG(value) as avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_temp desc;
