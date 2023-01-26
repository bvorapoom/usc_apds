select emp_no 
	from employees
	where lower(first_name) like '%mary%'
		and lower(last_name) like '%o_';
        
/* Output
+--------+
| emp_no |
+--------+
|  16021 |
|  21756 |
|  52983 |
|  73998 |
|  78783 |
|  88698 |
| 101753 |
| 216534 |
| 263268 |
| 410311 |
| 423386 |
| 459548 |
| 491899 |
+--------+
13 rows in set (0.15 sec)
*/