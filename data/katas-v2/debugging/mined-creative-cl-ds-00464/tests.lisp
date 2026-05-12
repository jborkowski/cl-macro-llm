;;; tests.lisp — generated from j14i/cl-ds row 464
;;; instruction: Write a Common Lisp macro `debug-print` that prints a labelled form and its result to *trace-output* and returns the res
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((debug-print "step1" (* x 2)) . (LET ((#:G1 (* X 2))) (FORMAT *TRACE-OUTPUT* "~&[~A] ~S = ~S~%" "step1" '(* X 2) #:G1) #:G1)))
