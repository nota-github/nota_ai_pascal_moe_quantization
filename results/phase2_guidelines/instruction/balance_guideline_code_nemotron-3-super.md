

1. Use descriptive, problem‑specific identifiers (e.g., `row_count`, `max_fruits`, `grid`, `ans`) instead of generic names like `i`, `j`, `tmp`, or `value`.  
2. Favor longer multi‑character tokens over single‑character tokens, because longer identifiers are less likely to trigger the frequently‑activated (blue) experts.  
3. Prefer domain‑specific language constructs (custom function names, meaningful variable names) and limit the use of generic Python keywords such as `def`, `return`, `for`, `if`, `while`, `range`, and `len`.  
4. Avoid over‑using built‑in functions and operators (`len`, `range`, `max`, `min`, `+`, `-`, `==`, etc.) by replacing them with variable‑based logic or helper functions whenever possible.  
5. Choose identifiers that are unlikely to appear in generic code (avoid common names like `data`, `value`, `temp`) and instead use terms that reflect the problem context (e.g., `basket`, `ans`, `left_fruit`).  
6. When writing loops, explicitly name the loop variable and avoid relying heavily on comprehensions or built‑in functions that increase blue‑token density.  
7. Increase the proportion of identifier tokens and decrease the proportion of punctuation/operator tokens in the generated samples, since punctuation and operators tend to be classified as blue.