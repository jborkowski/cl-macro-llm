;;; tests.lisp — generated from j14i/cl-ds row 421
;;; instruction: Write a Common Lisp macro `dbg-when` that evaluates an expression and returns its value; the expression and value are lo
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((dbg-when *debug* (compute)) . (LET ((#:G1 (COMPUTE))) (WHEN *DEBUG* (FORMAT *TRACE-OUTPUT* "~&DBG: ~S => ~S~%" '(COMPUTE) #:G1)) #:G1)))
