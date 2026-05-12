;;; tests.lisp — generated from j14i/cl-ds row 396
;;; instruction: Write a Common Lisp macro `tap-print` that evaluates a form once, prints a labeled message with its value to *trace-outp
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((tap-print "result" (+ 1 2)) . (LET ((#:G1 (+ 1 2))) (FORMAT *TRACE-OUTPUT* "~&[~A] ~S~%" "result" #:G1) #:G1)))
