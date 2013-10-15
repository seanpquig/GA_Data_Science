-- 1. What customers are frome the UK?
SELECT * FROM [Customers] WHERE Country = 'UK'

-- 2. What is the name of the customer who has the most orders?
SELECT CustomerName, COUNT(*) as order_count 
FROM Customers
INNER JOIN Orders on (Customers.CustomerID = Orders.CustomerID)
GROUP BY CustomerName
ORDER BY order_count DESC

-- 3. What supplier has the highest average product price?
SELECT SupplierName, AVG(Products.Price) as avg_price
FROM Suppliers
INNER JOIN Products on (Suppliers.SupplierID = Products.SupplierID)
GROUP BY SupplierName
ORDER BY avg_price DESC

-- 4. What category has the most orders?
SELECT CategoryName, COUNT(*) as order_count
FROM Categories
INNER JOIN Products on (Categories.CategoryID = Products.CategoryID)
INNER JOIN OrderDetails on (Products.ProductID = OrderDetails.ProductID)
GROUP BY CategoryName
ORDER BY order_count DESC

-- 5. What employee made the most sales (by number of sales)?
SELECT LastName, FirstName, COUNT(*) as order_count
FROM Employees
INNER JOIN Orders on (Employees.EmployeeID = Orders.EmployeeID)
GROUP BY LastName, FirstName
ORDER BY order_count DESC

-- 6. What employee made the most sales (by value of sales)?
SELECT LastName, FirstName, SUM(OrderDetails.Quantity*Products.Price) as sales_val
FROM Employees
INNER JOIN Orders on (Employees.EmployeeID = Orders.EmployeeID)
INNER JOIN OrderDetails on (Orders.OrderID = OrderDetails.OrderID)
INNER JOIN Products on (OrderDetails.ProductID = Products.ProductID)
GROUP BY LastName, FirstName
ORDER BY sales_val DESC

-- 7. What Employees have BS degrees? (Hint: Look at LIKE operator)
SElECT *
FROM Employees
WHERE Notes LIKE '%BS%'

-- 8. What supplier has the highest average product price 
--    assuming they have at least 2 products (Hint: Look at the HAVING operator)
Select SupplierName, AVG(Products.price) as avg_price
FROM Suppliers
INNER JOIN Products on (Suppliers.SupplierID = Products.SupplierID)
GROUP BY SupplierName
HAVING COUNT(Products.ProductID)>=2
ORDER BY avg_price DESC




