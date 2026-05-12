;;; tests.lisp — generated from j14i/cl-ds row 9
;;; instruction: Write a Common Lisp macro `with-trace` that prints an `Enter LABEL` line before the body and an `Exit LABEL` line after 
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-trace "compute" (heavy-work)) . (LET ((#:G1 "compute")) (FORMAT T "~&Enter ~A~%" #:G1) (UNWIND-PROTECT (PROGN (HEAVY-WORK)) (FORMAT T "~&Exit ~A~%" #:G1)))))
