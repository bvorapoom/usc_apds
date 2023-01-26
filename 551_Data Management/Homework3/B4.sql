 select distinct bar as Bar
	from Sells
    where price = (select max(price) from Sells);
    
/* Output
+-----------+
| Bar       |
+-----------+
| Joe's bar |
+-----------+
1 row in set (0.01 sec)
*/
 