;;; tests.lisp — generated from j14i/cl-ds row 75
;;; instruction: Write a Common Lisp macro `peek` that takes a label and an expression, prints the label and the evaluated value to *trac
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((peek "user-id" (get-user-id)) . (LET ((#:G1 (GET-USER-ID))) (FORMAT *TRACE-OUTPUT* "~&PEEK ~A: ~S~%" "user-id" #:G1) #:G1)))
