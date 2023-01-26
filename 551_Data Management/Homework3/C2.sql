select Manufacturer
	, avg(Price) as Average
	from Beers2Bars
    group by Manufacturer;
    
/* Output
+----------------+---------+
| Manufacturer   | Average |
+----------------+---------+
| Anheuser-Busch |       3 |
| Pete's         |     3.5 |
| Heineken       |       2 |
+----------------+---------+
3 rows in set (0.00 sec)
*/