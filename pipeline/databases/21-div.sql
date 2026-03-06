-- function SafeDiv that divides a/b or returns 0 if b is equal to 0.
DELIMITER $$

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    END IF;

RETURN a / b;
END $$
