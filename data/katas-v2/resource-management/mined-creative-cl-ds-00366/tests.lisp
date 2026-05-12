;;; tests.lisp — generated from j14i/cl-ds row 366
;;; instruction: Write a Common Lisp macro `with-restored-place` that captures the current value of a generalized place, runs the body (w
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-restored-place *config* (setf *config* :modified) (run)) . (LET ((#:G1 *CONFIG*)) (UNWIND-PROTECT (PROGN (SETF *CONFIG* :MODIFIED) (RUN)) (SETF *CONFIG* #:G1)))))
