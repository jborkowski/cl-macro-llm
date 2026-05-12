;;; tests.lisp — generated from j14i/cl-ds row 234
;;; instruction: Write a Common Lisp macro `dbg` that prints a form and its value to *trace-output*, then returns the value. Useful as a 
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((dbg (+ 1 2)) . (LET ((#:G1 (+ 1 2))) (FORMAT *TRACE-OUTPUT* "~&DBG: ~S = ~S~%" (QUOTE (+ 1 2)) #:G1) #:G1)))
