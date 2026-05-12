;;; tests.lisp — generated from j14i/cl-ds row 68
;;; instruction: Write a Common Lisp macro `assert-positive` that evaluates its argument once, signals an error if the value is not a pos
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-positive (- x 1)) . (LET ((#:G1 (- X 1))) (UNLESS (AND (NUMBERP #:G1) (PLUSP #:G1)) (ERROR "Expected positive number, got ~S" #:G1)) #:G1)))
