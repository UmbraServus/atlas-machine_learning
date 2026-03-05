-- trigger that resets the attribute valid_email when email has been changed.
DELIMITER $$

CREATE TRIGGER reset_attribute
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NEW.email <> OLD.email THEN
        SET  NEW.valid_email = 0;
    END IF;
END $$

DELIMITER ;