;;; tests.lisp — generated from j14i/cl-ds row 288
;;; instruction: Write a Common Lisp macro `bench-once` that runs an expression a single time and returns two values: the expression's va
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((bench-once (compute)) . (LET* ((#:G1 (GET-INTERNAL-REAL-TIME)) (#:G2 (COMPUTE)) (#:G3 (GET-INTERNAL-REAL-TIME))) (VALUES #:G2 (- #:G3 #:G1)))))
