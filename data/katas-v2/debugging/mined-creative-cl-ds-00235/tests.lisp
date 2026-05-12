;;; tests.lisp — generated from j14i/cl-ds row 235
;;; instruction: Write a Common Lisp macro `log-call` that invokes a function with the given arguments, prints a CALL line with the funct
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((log-call sqrt 16) . (LET ((#:G1 (SQRT 16))) (FORMAT *TRACE-OUTPUT* "~&CALL ~S~{ ~S~} => ~S~%" (QUOTE SQRT) (LIST 16) #:G1) #:G1)))
