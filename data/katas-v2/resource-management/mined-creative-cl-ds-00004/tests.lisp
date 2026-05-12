;;; tests.lisp — generated from j14i/cl-ds row 4
;;; instruction: Write a Common Lisp macro `with-rollback` that snapshots the value of a place, runs the body, and if the body exits non-
;;; category: resource-management | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((with-rollback *config* (modify *config*)) . (LET ((#:G1 *CONFIG*) (#:G2 NIL)) (UNWIND-PROTECT (MULTIPLE-VALUE-PROG1 (PROGN (MODIFY *CONFIG*)) (SETF #:G2 T)) (UNLESS #:G2 (SETF *CONFIG* #:G1))))))
