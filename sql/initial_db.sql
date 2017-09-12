CREATE DATABASE kaggle;

-- CREATE TABLE "titanic" --------------------------------------
CREATE TABLE "public"."titanic" (
	"passenger_id" Bigint NOT NULL,
	"survived" Boolean NOT NULL,
	"pclass" Integer NOT NULL,
	"name" Character Varying( 2044 ) NOT NULL,
	"sex" Character Varying( 2044 ) NOT NULL,
	"age" Double Precision,
	"sibsp" Integer,
	"parch" Integer,
	"ticket" Character Varying( 2044 ),
	"fare" Double Precision,
	"cabin" Character Varying( 2044 ),
	"embarked" Character Varying( 2044 ) );
 ;
