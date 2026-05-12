;;; tests.lisp — generated from j14i/cl-ds row 330
;;; instruction: Write a Common Lisp macro `with-saved-special` that captures the current value of a special variable, runs body (which m
;;; category: resource-management | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((with-saved-special *config* (setf *config* :new) (run)) . (LET ((#:G1 *CONFIG*)) (UNWIND-PROTECT (PROGN (SETF *CONFIG* :NEW) (RUN)) (SETF *CONFIG* #:G1)))))
