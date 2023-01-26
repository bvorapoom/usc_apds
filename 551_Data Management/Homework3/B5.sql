select drinker as Drinker
	from Likes l1
    where l1.beer = 'Bud'
		and not exists (select drinker from Likes l2 where l1.drinker = l2.drinker and l2.beer = 'Summerbrew') ;
        
/* Output
+----------+
| Drinker  |
+----------+
| Bill     |
| Jennifer |
+----------+
2 rows in set (0.00 sec)
*/

