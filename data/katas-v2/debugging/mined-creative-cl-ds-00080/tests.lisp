;;; tests.lisp — generated from j14i/cl-ds row 80
;;; instruction: Write a Common Lisp macro `logged-block` that prints an ENTER message before running a body and an EXIT message with the
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((logged-block "main" (process)) . (PROGN (FORMAT *TRACE-OUTPUT* "~&ENTER ~A~%" "main") (LET ((#:G1 (PROGN (PROCESS)))) (FORMAT *TRACE-OUTPUT* "~&EXIT ~A => ~S~%" "main" #:G1) #:G1))))
