;;; tests.lisp — generated from j14i/cl-ds row 172
;;; instruction: Write a Common Lisp macro `trace-block` that prints ENTER and EXIT markers around a body of code (with a user-supplied l
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((trace-block "foo" (compute)) . (PROGN (FORMAT *TRACE-OUTPUT* "~&[ENTER ~A]~%" "foo") (LET ((#:G1 (PROGN (COMPUTE)))) (FORMAT *TRACE-OUTPUT* "~&[EXIT ~A]~%" "foo") #:G1))))
