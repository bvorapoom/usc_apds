select d.name as Drinker
	from Drinkers d
    left join Frequents f
		on d.name = f.drinker
	where f.drinker is null;
    
/* Output
Empty set (0.00 sec)
*/

