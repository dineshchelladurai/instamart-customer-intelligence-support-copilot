-- Q1: WHERE then HAVING; completed orders only.
SELECT customer_id, SUM(order_amount) AS spend_over_500
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: revenue by city and product category.
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: products priced above the catalogue average.
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers without any completed order.
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: customer completed spend compared with the average customer spend.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed'
    GROUP BY customer_id
), average_spend AS (
    SELECT AVG(total_spend) AS mean_spend FROM customer_spend
)
SELECT customer_id, total_spend
FROM customer_spend
WHERE total_spend > (SELECT mean_spend FROM average_spend);

-- Q6: city-local rankings of completed spend.
WITH customer_spend AS (
    SELECT c.city, c.customer_id, SUM(o.order_amount) AS total_spend
    FROM customers c JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city, customer_id, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_number
FROM customer_spend;

-- Q7: daily completed revenue and cumulative revenue.
WITH daily_revenue AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
