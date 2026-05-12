;;; tests.lisp — generated from j14i/cl-ds row 42
;;; instruction: Write a Common Lisp macro `signum-of` that expands to a form returning -1, 0, or 1 according to the sign of its argument
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((signum-of (read-temperature)) . (LET ((#:G1 (READ-TEMPERATURE))) (COND ((ZEROP #:G1) 0) ((PLUSP #:G1) 1) (T -1)))))
