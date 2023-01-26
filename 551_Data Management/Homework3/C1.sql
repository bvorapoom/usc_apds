DROP VIEW if exists Beers2Bars;

CREATE VIEW Beers2Bars AS
(
	select manf as Manufacturer
		, beer as Beer
		, bar as Bar
		, price as Price
		from Sells s
		left join Beers b 
			on s.beer = b.name
);

/* Output
Query OK, 0 rows affected (0.01 sec)
*/