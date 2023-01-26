select e.first_name
	, e.last_name
    , t.title
    from employees e
    left join titles t
		on e.emp_no = t.emp_no
	where lower(t.title) like '%engineer%'
		and t.from_date = '2000-3-23';
        
/* Output
+------------+---------------+-----------------+
| first_name | last_name     | title           |
+------------+---------------+-----------------+
| Nahla      | Silva         | Engineer        |
| Uli        | Lichtner      | Senior Engineer |
| Matk       | Schlegelmilch | Senior Engineer |
| Mayuko     | Luff          | Engineer        |
| Masami     | Panienski     | Senior Engineer |
| Tzvetan    | Brodie        | Senior Engineer |
| Kerhong    | Pappas        | Senior Engineer |
| Xiaoshan   | Keirsey       | Senior Engineer |
| Jiakeng    | Baaleh        | Senior Engineer |
| Fox        | Begiun        | Senior Engineer |
+------------+---------------+-----------------+
10 rows in set (0.17 sec)
*/