;;; tests.lisp — generated from j14i/cl-ds row 142
;;; instruction: Write a Common Lisp macro `if-pred` that evaluates a value expression once into a hidden gensym, applies a predicate to 
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((if-pred numberp x n (+ n 1) :nope) . (LET ((#:G1 X)) (IF (NUMBERP #:G1) (LET ((N #:G1)) (+ N 1)) :NOPE))))
