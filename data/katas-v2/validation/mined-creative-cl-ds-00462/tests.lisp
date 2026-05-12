;;; tests.lisp — generated from j14i/cl-ds row 462
;;; instruction: Write a Common Lisp macro `assert-all-plusp` that takes any number of expressions, evaluates each once, and signals an e
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-all-plusp x y) . (PROGN (LET ((#:G1 X)) (UNLESS (PLUSP #:G1) (ERROR "Expected positive, got ~A from ~S" #:G1 'X))) (LET ((#:G2 Y)) (UNLESS (PLUSP #:G2) (ERROR "Expected positive, got ~A from ~S" #:G2 'Y))))))
