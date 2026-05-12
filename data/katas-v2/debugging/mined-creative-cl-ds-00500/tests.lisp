;;; tests.lisp — generated from j14i/cl-ds row 500
;;; instruction: Write a Common Lisp macro `spy` that wraps an expression, prints the source form and its evaluated result to standard ou
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((spy (+ 1 2)) . (LET ((#:G1 (+ 1 2))) (FORMAT T "~&SPY ~S => ~S~%" '(+ 1 2) #:G1) #:G1)))
