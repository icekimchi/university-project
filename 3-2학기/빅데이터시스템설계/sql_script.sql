SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `mydb` DEFAULT CHARACTER SET utf8 ;
USE `mydb` ;

-- -----------------------------------------------------
-- Table `mydb`.`user`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`user` (
  `u_id` VARCHAR(25) NOT NULL,
  `u_pw` VARCHAR(25) NOT NULL,
  `u_name` VARCHAR(25) NOT NULL,
  `u_email` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`u_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`place`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`place` (
  `p_id` VARCHAR(25) NOT NULL,
  `add_1` VARCHAR(45) NULL,
  `add_2` VARCHAR(45) NULL,
  `lat` VARCHAR(25) NULL,
  `lot` VARCHAR(25) NULL,
  PRIMARY KEY (`p_id`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `mydb`.`card`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`card` (
  `c_id` INT NOT NULL AUTO_INCREMENT,
  `c_name` VARCHAR(25) NOT NULL,
  `c_num` VARCHAR(45) NOT NULL,
  `c_CVC` VARCHAR(10) NOT NULL,
  `c_valid` VARCHAR(25) NOT NULL,
  `user_u_id` VARCHAR(25) NOT NULL,
  PRIMARY KEY (`c_id`),
  INDEX `fk_card_user_idx` (`user_u_id` ASC) VISIBLE,
  CONSTRAINT `fk_card_user`
    FOREIGN KEY (`user_u_id`)
    REFERENCES `mydb`.`user` (`u_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`bicycle`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`bicycle` (
  `b_id` INT NOT NULL AUTO_INCREMENT,
  `place_p_id` VARCHAR(25) NULL,
  PRIMARY KEY (`b_id`),
  INDEX `fk_bicycle_place1_idx` (`place_p_id` ASC) VISIBLE,
  CONSTRAINT `fk_bicycle_place1`
    FOREIGN KEY (`place_p_id`)
    REFERENCES `mydb`.`place` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`History`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`History` (
  `h_id` INT NOT NULL AUTO_INCREMENT,
  `h_price` INT NOT NULL,
  `s_time` DATETIME NOT NULL,
  `e_time` DATETIME NOT NULL,
  `bicycle_b_id` INT NOT NULL,
  `user_u_id` VARCHAR(25) NOT NULL,
  `s_place` VARCHAR(25) NOT NULL,
  `e_place` VARCHAR(25) NOT NULL,
  PRIMARY KEY (`h_id`),
  INDEX `fk_History_bicycle1_idx` (`bicycle_b_id` ASC) VISIBLE,
  INDEX `fk_History_user1_idx` (`user_u_id` ASC) VISIBLE,
  INDEX `fk_History_place1_idx` (`s_place` ASC) VISIBLE,
  INDEX `fk_History_place2_idx` (`e_place` ASC) VISIBLE,
  CONSTRAINT `fk_History_bicycle1`
    FOREIGN KEY (`bicycle_b_id`)
    REFERENCES `mydb`.`bicycle` (`b_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_History_user1`
    FOREIGN KEY (`user_u_id`)
    REFERENCES `mydb`.`user` (`u_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_History_place1`
    FOREIGN KEY (`s_place`)
    REFERENCES `mydb`.`place` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_History_place2`
    FOREIGN KEY (`e_place`)
    REFERENCES `mydb`.`place` (`p_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
