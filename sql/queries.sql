-- 1. Total gross revenue in completed orders
SELECT ROUND(SUM(order_amount), 2) AS revenue_completed FROM orders WHERE status = 'completed';

-- 2. Order completion rate by city
SELECT c.city, COUNT(o.order_id) AS orders, SUM(CASE WHEN o.status = 'completed' THEN 1 ELSE 0 END) AS completed
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY completed DESC;

-- 3. Average order value by product category
SELECT p.category, ROUND(AVG(o.order_amount), 2) AS avg_order_value
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.category;

-- 4. Most common late-delivery product categories
SELECT p.category, COUNT(*) AS late_orders
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed' AND o.delivery_minutes > 45
GROUP BY p.category
ORDER BY late_orders DESC;

-- 5. Top customers by lifetime value
SELECT c.customer_id, ROUND(SUM(o.order_amount), 2) AS lifetime_value
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY lifetime_value DESC
LIMIT 10;

-- 6. New customers by month
SELECT strftime('%Y-%m', signup_date) AS signup_month, COUNT(*) AS customers
FROM customers
GROUP BY signup_month
ORDER BY signup_month;

-- 7. Cancelled-order share by city
SELECT c.city, ROUND(100.0 * SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(o.order_id), 2) AS cancelled_pct
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY cancelled_pct DESC;
