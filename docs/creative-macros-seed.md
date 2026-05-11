# Creative macros — seed list for `j14i/cl-macros-creative`

30 macro ideas spanning categories cl-ds undersamples. Hand-verified
later in SBCL before pushing to HF.

Format mirrors cl-ds schema: `instruction` is the prompt the model
receives; `input` is a sample call; `output` is the reference defmacro;
`macroexpand` is the verified expansion of the call.

Marked **★** are highest-leverage for GRPO — non-trivial mechanics
(gensym, once-only, multi-binding, scope discipline) and not in any
standard library.

---

## Tier 1 — Pure CL, broadly useful, easy to verify

### 1. `with-stopwatch` ★ (already verified)

**instruction:** Measure wall-clock elapsed time of a body. Return two
values: elapsed seconds (float) and the body's last value.
**input:** `(with-stopwatch (sleep 0.1) (* 6 7))`
**output:**
```lisp
(defmacro with-stopwatch (&body body)
  (let ((start (gensym)) (end (gensym)) (result (gensym)))
    `(let ((,start (get-internal-real-time)))
       (let ((,result (progn ,@body)))
         (let ((,end (get-internal-real-time)))
           (values (float (/ (- ,end ,start) internal-time-units-per-second))
                   ,result))))))
```
**category:** efficiency | **technique:** gensym | **complexity:** intermediate

### 2. `when-let`

**instruction:** Bind a value to a name, run body only if value is non-nil.
**input:** `(when-let (x (find-thing)) (use x))`
**output:**
```lisp
(defmacro when-let ((name expr) &body body)
  `(let ((,name ,expr))
     (when ,name ,@body)))
```

### 3. `if-let`

**instruction:** Bind a value to a name; if non-nil run then-form with the
binding in scope, else run else-form (no binding).
**input:** `(if-let (x (lookup k)) (process x) :not-found)`

### 4. `awhen` (anaphoric when)

**instruction:** Like when, but bind the test value to `it` inside body.
**input:** `(awhen (find-foo) (process it))`

### 5. `aif` (anaphoric if)

**instruction:** Like if, but bind the test value to `it` inside both branches.
**input:** `(aif (compute) (use it) :no-result)`

---

## Tier 2 — Resource management

### 6. `with-retry` ★

**instruction:** Run body; if it signals an error, retry up to N times
with exponential backoff (base sleep doubles each attempt). Re-signal
the last error if all attempts fail.
**input:** `(with-retry (:max-attempts 3 :base-sleep 0.5) (flaky-call))`
**output:** (sketch)
```lisp
(defmacro with-retry ((&key (max-attempts 3) (base-sleep 0.1)) &body body)
  (let ((attempt (gensym)) (sleep-time (gensym)) (last-err (gensym)))
    `(let ((,attempt 0) (,sleep-time ,base-sleep) ,last-err)
       (loop while (< ,attempt ,max-attempts) do
         (handler-case (return (progn ,@body))
           (error (c)
             (setf ,last-err c)
             (incf ,attempt)
             (when (< ,attempt ,max-attempts)
               (sleep ,sleep-time)
               (setf ,sleep-time (* ,sleep-time 2)))))
         finally (error ,last-err)))))
```

### 7. `with-temp-file` ★

**instruction:** Create a temporary file, bind its pathname to a name,
execute body, delete the file on exit (even on error). Suffix optional.
**input:** `(with-temp-file (path :suffix ".lisp") (write-thing path) (parse-thing path))`

### 8. `with-cleanup` ★

**instruction:** Execute body, then always run cleanup forms (like
unwind-protect, but more readable for multiple cleanup steps).
**input:**
```
(with-cleanup ((close-conn db) (free-buffer buf))
  (use db) (use buf))
```

### 9. `with-rollback`

**instruction:** Execute body. If body completes normally, return its value.
If body signals any error, run the rollback forms before re-signalling.
**input:** `(with-rollback ((revert-state)) (mutate-state) (continue))`

### 10. `with-mutex-held`

**instruction:** Acquire a mutex, run body, release mutex even on error.
Assumes `acquire-mutex` and `release-mutex` are available (sb-thread or bordeaux).
**input:** `(with-mutex-held (my-lock) (critical-section))`

---

## Tier 3 — Binding extensions

### 11. `and-let*` ★

**instruction:** Sequential binding form that short-circuits if any
binding evaluates to nil. Returns nil if any binding is nil, otherwise
runs body.
**input:**
```
(and-let* ((x (lookup k))
           (y (compute x))
           ((plusp y)))
  (use y))
```

### 12. `let-with-defaults`

**instruction:** Bind variables, using a default form if the value
expression evaluates to nil.
**input:**
```
(let-with-defaults ((x (env-var "X") :default "fallback")
                    (y (env-var "Y") :default 42))
  (list x y))
```

### 13. `lazy-let` ★

**instruction:** Bindings evaluate lazily — the value-form runs only
the first time the variable is referenced, and is memoized thereafter.
**input:** `(lazy-let ((x (expensive))) (when condition (use x)))`

### 14. `bind-or`

**instruction:** Bind name to the first non-nil value among a list of
expressions; run body with that binding.
**input:**
```
(bind-or (x (env-var "FOO") (config-key :foo) "default")
  (use x))
```

### 15. `destructure-as` ★

**instruction:** Pattern-match binding for plists, alists, or
structs. Like destructuring-bind but selects the binding strategy from
a keyword.
**input:**
```
(destructure-as :plist ((name :name) (age :age :default 0)) plist
  (format nil "~A ~A" name age))
```

---

## Tier 4 — Caching / memoization

### 16. `defmemo` ★

**instruction:** Define a function that memoizes its result by EQUAL on
arguments. Cache cleared with `(clear-memo 'name)`.
**input:** `(defmemo fib (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))`

### 17. `with-cache` ★

**instruction:** Lexical scope holding a memoization cache for one
function call site. Cleared on exit from body.
**input:**
```
(with-cache
  (loop for x in xs collect (expensive x)))
```
(expensive is memoized only within the body, not globally)

### 18. `memoize`

**instruction:** Wrap a function (passed as a symbol) with memoization,
mutating the symbol-function. Idempotent (safe to call twice).
**input:** `(memoize 'fib)`

---

## Tier 5 — Iteration / accumulation

### 19. `dohash` ★

**instruction:** Iterate over a hash table, binding key and value to
names per iteration. Body sees both bindings.
**input:** `(dohash (k v my-table) (format t "~A => ~A~%" k v))`

### 20. `doplist`

**instruction:** Iterate over a property list two-at-a-time, binding
key and value per pair.
**input:** `(doplist (k v '(:a 1 :b 2)) (format t "~A=~A " k v))`

### 21. `with-collected` ★

**instruction:** Within body, the function `collect` accumulates values
into a result list. Returns the list at the end.
**input:**
```
(with-collected
  (dotimes (i 5)
    (when (oddp i) (collect (* i i)))))
;; -> (1 9 25)
```

### 22. `dotimes-collecting`

**instruction:** Like dotimes but accumulates the body's value each
iteration into a list and returns it.
**input:** `(dotimes-collecting (i 5) (* i i))` → `(0 1 4 9 16)`

---

## Tier 6 — Threading / composition

### 23. `->` (thread-first)

**instruction:** Clojure-style thread-first: insert each form's previous
value as the *second* element (the first argument).
**input:** `(-> "Hello" (concatenate 'string " world") (string-upcase))`
→ `(STRING-UPCASE (CONCATENATE 'STRING "Hello" " world"))`

### 24. `->>` (thread-last)

**instruction:** Same but insert as the last argument.
**input:** `(->> '(1 2 3 4) (mapcar #'1+) (remove-if-not #'oddp))`

### 25. `as->` ★

**instruction:** Threading macro that binds the intermediate to a
user-chosen name (works when arg position varies per call).
**input:**
```
(as-> '(1 2 3) it
  (mapcar #'1+ it)
  (cons 0 it)
  (length it))
```

---

## Tier 7 — Inspection / instrumentation

### 26. `with-instrumentation` ★

**instruction:** Wrap body with timing + logging. Print start, end,
elapsed time, and any signalled condition.
**input:** `(with-instrumentation (:label "load") (load-data))`

### 27. `inline-test` ★

**instruction:** Inside a defun body, declare an inline test that runs
at compile time and signals an error if it fails. The test does NOT run
when the function is called.
**input:**
```
(defun fact (n)
  (inline-test (= 6 (fact 3)))
  (if (zerop n) 1 (* n (fact (- n 1)))))
```

### 28. `with-trace`

**instruction:** Within body, every call to functions in the given list
is traced (entry/exit/value). Untrace on exit.
**input:** `(with-trace (foo bar) (do-thing))`

---

## Tier 8 — State / control

### 29. `defstate` ★

**instruction:** Define a finite state machine. Each state has an
action and a transitions table; the macro produces both a struct and
an advance function.
**input:**
```
(defstate door
  (closed :on :open  -> opening)
  (opening :on :stop -> ajar  :on :done -> open)
  (open    :on :close -> closing))
```

### 30. `define-condition+`

**instruction:** Define a condition with auto-generated `:report`,
slot accessors with sensible names, and a constructor.
**input:**
```
(define-condition+ db-error (error)
  ((sql :type string)
   (host :type string))
  :report "DB error on ~A executing: ~A" host sql)
```

---

## Authoring workflow

1. Pick 25-30 from this list (or replace any with project-specific ones)
2. For each: I write the full implementation + verified expansion
3. Hand-test each in SBCL, drop ones that don't pass
4. Format as JSONL row matching cl-ds schema
5. Push to `j14i/cl-macros-creative`
6. Run `cl_ds_to_katas.py` against it
7. Expect ≥ 90% keep rate (these are hand-verified, not dataset-noise)

## Which to do first

If you want me to write full implementations + verified expansions for
the next iteration, prioritize the **★ marked** ones — they're the
mechanics-heavy macros that exercise gensym discipline, once-only
evaluation, multi-binding scope, condition handling. The non-starred
ones are simpler restatements of well-known patterns; useful but
lower-leverage.

**Suggested first cut: 15 macros = the 12 starred + 3 simple ones for
warmup (when-let, awhen, ->).**

Tell me which ones, I'll write implementations + sample expansions, and
we'll verify each in SBCL on your Mac before pushing to HF.
