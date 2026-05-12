;;; tests.lisp — generated from j14i/cl-ds row 528
;;; instruction: Write a Common Lisp macro `trace-form` that evaluates a form, prints the form and its result to *trace-output*, and retu
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((trace-form (+ 1 2)) . (LET ((#:G1 (+ 1 2))) (FORMAT *TRACE-OUTPUT* "~&~S => ~S~%" '(+ 1 2) #:G1) #:G1)))
