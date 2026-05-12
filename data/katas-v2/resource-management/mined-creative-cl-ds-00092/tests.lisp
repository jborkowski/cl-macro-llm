;;; tests.lisp — generated from j14i/cl-ds row 92
;;; instruction: Write a Common Lisp macro `with-shadowed-symbols` that saves the current values of a list of special variables to gensym
;;; category: resource-management | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((with-shadowed-symbols (*x*) (setf *x* 42) (work)) . (LET ((#:G1 *X*)) (UNWIND-PROTECT (PROGN (SETF *X* 42) (WORK)) (SETF *X* #:G1)))))
