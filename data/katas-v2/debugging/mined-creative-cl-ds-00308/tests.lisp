;;; tests.lisp — generated from j14i/cl-ds row 308
;;; instruction: Write a Common Lisp macro `with-instrumentation` that prints an entry message before body executes and an exit message w
;;; category: debugging | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((with-instrumentation "step1" (do-work)) . (LET ((#:G1 "step1")) (FORMAT *TRACE-OUTPUT* "~&ENTER ~A~%" #:G1) (LET ((#:G2 (PROGN (DO-WORK)))) (FORMAT *TRACE-OUTPUT* "~&EXIT ~A => ~S~%" #:G1 #:G2) #:G2))))
