;;; tests.lisp — generated from j14i/cl-ds row 349
;;; instruction: Write a Common Lisp macro `cube-and-square` that evaluates an expression once and binds its cube and square to user-name
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((cube-and-square (c s) (compute) (list c s)) . (LET ((#:G1 (COMPUTE))) (LET ((C (* #:G1 #:G1 #:G1)) (S (* #:G1 #:G1))) (LIST C S)))))
